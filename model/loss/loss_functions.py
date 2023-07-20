import whisper
import torch
import torchaudio
from model.datasets import *
from model.datasets import torch
from model.loss import moving_average
import torch.nn.functional as F

class Loss(torch.nn.Module):
    def __init__(self, name, weight) -> None:
        super().__init__()
        self.name = name
        self.moving_average = moving_average.EMA(100)
        
    def get_value(self, *args):
        raw = self.get_raw_value(*args)
        return raw / (self.moving_average.update(raw) * 0.999)  # TODO should be applied b4 or after
    
    def forward(self, *args):
        return self.get_value(*args)

    def get_raw_value(self, *args) -> torch.Tensor:
        raise NotImplemented
    

class ReconstructionLoss(Loss):
    def __init__(self, weight, beta=1) -> None:
        super().__init__("spec1 loss", weight)
        self.specs = torch.nn.ModuleList([torchaudio.transforms.MelSpectrogram(16000, n_mels=64, n_fft=2 ** (i+1), win_length=2**i, hop_length=2 ** (i-2), f_max=8000, f_min=0) for i in range(5, 12)])  #  see if the values here are reasonable, could well be way off
        self.beta = beta

    def loss_for_spec(self, x, y, spec):
        xs, ys = spec(x), spec(y)
        return F.l1_loss(xs, ys) + self.beta * F.mse_loss(xs, ys)

    def get_raw_value(self, x, y):
        total = 0.0
        for spec in self.specs:
            total = total + self.loss_for_spec(x, y, spec)
        return total


class SetLoss(Loss):
    def get_raw_value(self, raw_value):
        return raw_value


class WhisperMel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160)
        self.mel_filters = torchaudio.functional.melscale_fbanks(n_freqs=201, f_max=8000, f_min=0, n_mels=80, sample_rate=16000, norm="slaney", mel_scale="slaney")

    def forward(self, x):
        # steps here taken from whisper.log_mel_spectrogram in order to maintain compatability (not have to retrain whisper), but this is much faster
        spec = self.spectrogram(x)[..., :-1]
        mel = (spec.transpose(-1, -2) @ self.mel_filters).transpose(-1, -2)

        log = torch.clamp_min(mel, 1e-10).log10() # clamp to prevent divide by 0
        log = torch.maximum(log, log.max() - 8.0)
        log = (log + 4.0) / 4.0
        return log
    
class WhisperLoss(Loss):
    def __init__(self, context_length, batch_size, weight, beta=1.) -> None:
        super().__init__("Whisper Loss", weight)
        assert context_length == 240*48 and batch_size == 64, "TODO: not implemented variable batch and context size"
        self.beta = beta
        self.padding = 1500
        self.batch_size = batch_size
        self.full_length = context_length + 2 * self.padding
        
        self.whisper = whisper.load_model("tiny.en")
        self.melspec = WhisperMel()
    
    def get_intermediate(self, x):
        return self.whisper.encoder(x)
    
    def process_batch(self, x):
        x = torch.nn.functional.pad(x, (self.padding, self.padding))

        if x.shape[0] != self.batch_size:  # make up the dimensions to 
            t = torch.zeros(self.batch_size, 1, self.full_length).to(x.device)
            t[:x.shape[0], ...] = x
            x = t

        return x.view(-1, 480000)  
    
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

    def get_raw_value(self, discrim_y) -> torch.Tensor:
        values = torch.maximum(1-discrim_y, torch.tensor([0.]))
        return torch.mean(values)


class DiscriminatorAdversairialLoss(Loss):
    def __init__(self, weight) -> None:
        super().__init__("Discriminator Adversairial Loss", weight)

    def get_value(self, discrim_x, discrim_y) -> torch.Tensor:  # get value as we don't want to apply weight balancing
        xs = torch.maximum(1-discrim_x, torch.tensor([0.]))
        ys = torch.maximum(1+discrim_y, torch.tensor([0.]))
        return torch.mean(xs) + torch.mean(ys)