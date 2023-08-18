import whisper
import torch
import torchaudio
from model.datasets import *
from model.datasets import torch
from model.loss import moving_average
import torch.nn.functional as F

class Loss(torch.nn.Module):
    def __init__(self, name, weight, normalize=True) -> None:
        super().__init__()
        self.name = name
        self.moving_average = moving_average.EMA(1000, beta=0.99)
        self.plot_average = moving_average.SMA(25)  # should always be the same as the plot interval, need to figure out a way to make this so
        self.prev_raw = -1
        self.previous = []
        self.weight = weight
        self.normalize = normalize
        
    def get_value(self, *args):
        raw = self.get_raw_value(*args)
        self.prev_raw = raw
        self.plot_average.update(raw)
        #return raw
        if self.normalize:
            return self.weight * (raw / (self.moving_average.update(raw.item()) * 0.999))  # TODO should be applied b4 or after
        else:
            return self.weight * raw
    
    def forward(self, *args):
        return self.get_value(*args)

    def get_raw_value(self, *args) -> torch.Tensor:
        raise NotImplemented
    
    def plot(self, writer, steps):
        writer.add_scalar(f"train_loss/{self.name}", self.prev_raw, steps)
    
    

class ReconstructionLossFreq(Loss):
    def __init__(self, weight, beta=1) -> None:
        super().__init__("frequency loss", weight)
        self.specs = torch.nn.ModuleList([torchaudio.transforms.MelSpectrogram(16000, n_mels=80, n_fft=2 ** (i+1), win_length=2**i, hop_length=2 ** (i-2), f_max=8000, f_min=0) for i in range(9, 10)])  #  see if the values here are reasonable, could well be way off
        self.beta = beta

    def loss_for_spec(self, x, y, spec):
        xs, ys = spec(x), spec(y)
        return F.l1_loss(xs, ys) + self.beta * F.mse_loss(xs, ys)  # TODO soundstream seems to take a log here, could fix the problem of mse being so much higher than l1

    def get_raw_value(self, x, y):
        total = 0.0
        for spec in self.specs:
            total = total + self.loss_for_spec(x, y, spec)
        return total / len(self.specs)
    
class ReconstructionLossTime(Loss):  # From what I can tell the ONLY purpose of this is making silence actually silent, some noise can escape the freq loss.
    def __init__(self, weight, **args) -> None:
        super().__init__("time loss", weight, **args)

    def get_raw_value(self, x, y):
        return torch.nn.functional.l1_loss(x, y)
    

class ReconstructionLoss(Loss):
    def __init__(self, time_weight, freq_weight, beta=1) -> None:
        super().__init__("Reconstruction Loss", 1, normalize=False)
        self.time_factor = ReconstructionLossTime(time_weight)
        self.freq_factor = ReconstructionLossFreq(freq_weight)

    def get_raw_value(self, x, y):
        #return self.loss_for_spec(x, y, self.spec) / (750 * 4 * 4)
        return self.time_factor(x, y) + self.freq_factor(x, y)

    def plot(self, writer, steps):
        self.time_factor.plot(writer, steps)
        self.freq_factor.plot(writer, steps)

class SetLoss(Loss):
    def get_raw_value(self, raw_value):
        return raw_value


class WhisperMel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160)
        self.register_buffer("mel_filters", torchaudio.functional.melscale_fbanks(n_freqs=201, f_max=8000, f_min=0, n_mels=80, sample_rate=16000, norm="slaney", mel_scale="slaney"))

    def forward(self, x):
        # steps here taken from whisper.log_mel_spectrogram in order to maintain compatability (not have to retrain whisper), but this is much faster
        spec = self.spectrogram(x)[..., :-1]
        mel = (spec.transpose(-1, -2) @ self.mel_filters).transpose(-1, -2)

        log = torch.clamp_min(mel, 1e-10).log10() # clamp to prevent divide by 0
        log = torch.maximum(log, log.max() - 8.0)
        log = (log + 4.0) / 4.0
        return log
    

class WhisperLoss(Loss):
    WHISPER_EXPECTED = 480000
    WHISPER_EXPECTED_FACTORS = [7500, 8000, 9600, 10000, 12000, 15000, 16000, 19200, 20000, 24000, 30000, 32000, 40000, 48000, 60000, 80000, 96000, 120000, 160000, 240000, 480000]  # precomputed for performance, low ommitted as they will never be used
    MIN_PADDING = 2000

    def __init__(self, context_length, batch_size, weight, beta=1.) -> None:
        super().__init__("Whisper Loss", weight)
        self.beta = beta
        self.batch_size = batch_size
        self.ctx_len = context_length

        self.wanted_bsize, self.padding, self.target_length = self.calc_operations(context_length, batch_size)  #  TODO sort this line out
        print(self.wanted_bsize, self.padding, self.target_length)
        
        self.whisper = whisper.load_model("tiny.en")
        self.melspec = WhisperMel()

    def calc_operations(self, ctx_len, batch_size):
        candidates = [i for i in WhisperLoss.WHISPER_EXPECTED_FACTORS if i > ctx_len + WhisperLoss.MIN_PADDING]
        aimed_length = candidates[0]  # just take the first, will be the most efficient
        padding = aimed_length - ctx_len

        # stage 2, find the wanted batch size (we will add padding until the required batch size = wanted)
        amount_per_batch = int(WhisperLoss.WHISPER_EXPECTED / aimed_length)
        whisper_batch_size = math.ceil(batch_size / amount_per_batch)
        wanted_batch_size = whisper_batch_size * amount_per_batch
        return (wanted_batch_size, padding, aimed_length)

    
    def get_intermediate(self, x):
        return self.whisper.encoder(x)
    
    def process_batch(self, x):
        x = torch.nn.functional.pad(x, (0, self.padding))

        if x.shape[0] != self.wanted_bsize:  # make up the dimensions to 
            t = torch.zeros(self.wanted_bsize, 1, self.target_length).to(x.device)
            t[:x.shape[0], ...] = x
            x = t

        return x.view(-1, WhisperLoss.WHISPER_EXPECTED)  
    
    def get_raw_value(self, x, y) -> torch.Tensor:
        if self.training:
            xp, yp = self.process_batch(x), self.process_batch(y)
        else:
            xp, yp = torch.squeeze(x, 1), torch.squeeze(y, 1)

        xspec, yspec = self.melspec(xp), self.melspec(yp)
        xw, yw = self.get_intermediate(xspec), self.get_intermediate(yspec)

        return F.l1_loss(xw, yw) + self.beta * F.mse_loss(xw, yw)
    

class DiscriminatorLoss(Loss):
    def __init__(self, weight) -> None:
        super().__init__("Discriminator Loss", weight)
        self.register_buffer("empty", torch.tensor([0.]))

    def get_raw_value(self, discrim_y) -> torch.Tensor:
        values = torch.maximum(1-discrim_y, self.empty)
        return torch.mean(values)


class DiscriminatorAdversairialLoss(Loss):
    def __init__(self, weight) -> None:
        super().__init__("Discriminator Adversairial Loss", weight, normalize=False)
        self.register_buffer("empty", torch.tensor([0.]))

    def get_raw_value(self, discrim_x, discrim_y) -> torch.Tensor:  # get value as we don't want to apply weight balancing
        xs = torch.maximum(1-discrim_x, self.empty)
        ys = torch.maximum(1+discrim_y, self.empty)
        l = torch.mean(xs) + torch.mean(ys)
        self.plot_average.update(l)  # keep updating moving average anyway for logging purposes
        return l
    

class FeatureLoss(Loss):
    def __init__(self, weight) -> None:
        super().__init__("Discriminator Feature Loss", weight)

    def get_raw_value(self, feat_x: list[list[torch.Tensor]], feat_y: list[list[torch.Tensor]]) -> torch.Tensor:  # lists are both of the form discim, layer, tensor<b, x, y>
        total = 0.
        assert len(feat_x) == len(feat_y)  # these dims should be the same, not prod code either so better to hard crash

        for disc_x, disc_y in zip(feat_x, feat_y):
            assert len(disc_x) == len(disc_y)  # these dims should be the same, not prod code either so better to hard crash
            disc_total = 0.
            for l_x, l_y in zip(disc_x, disc_y):
                disc_total += torch.nn.functional.l1_loss(l_x, l_y)

            total += disc_total / len(disc_x)     
        return total / len(feat_x) # the above is just torch.nn.functional.l1_loss(feat_x, feat_y), but we're using lists for some stuff not tensors so have to do it manually