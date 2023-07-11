import whisper
import torch
import torchaudio
from model.datasets import *

class CustomMel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160)
        self.mel_filters = torchaudio.functional.melscale_fbanks(n_freqs=201, f_max=8000, f_min=0, n_mels=80, sample_rate=16000, norm="slaney", mel_scale="slaney")

    def forward(self, x):
        # steps here taken from whisper.log_mel_spectrogram in order to maintain compatability, but this is much faster
        spec = self.spectrogram(x)[..., :-1]
        mel = (spec.transpose(-1, -2) @ self.mel_filters).transpose(-1, -2)

        log = torch.clamp_min(mel, 1e-10).log10() # clamp to prevent divide by 0
        log = torch.maximum(log, log.max() - 8.0)
        log = (log + 4.0) / 4.0
        return log
    
class WhisperLoss(torch.nn.Module):
    def __init__(self, context_length, batch_size) -> None:
        super().__init__()
        assert context_length == 240*40 and batch_size == 64, "TODO: not implemented variable batch and context size"
        self.padding = 2700
        self.batch_size = batch_size
        self.full_length = context_length + 2 * self.padding
        
        self.model = whisper.load_model("tiny.en")
        self.device = "cpu"

    def to(self, d):
        super().to(d)
        self.device = d
        return self
    
    def forward(self, x):
        return self.model.encoder(x)
    
    def process_batch(self, x):
        x = torch.nn.functional.pad(x, (self.padding, self.padding))

        if x.shape[0] != self.batch_size:  # make up the dimensions to 
            t = torch.zeros(self.batch_size, 1, self.full_length).to(self.device)
            t[:x.shape[0], ...] = x
            x = t

        return x.view(-1, 480000)  
