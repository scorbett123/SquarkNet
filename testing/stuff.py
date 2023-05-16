import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import random
import matplotlib.pyplot as plt
import utils
import einops

class SpeechDataset(Dataset):
    def __init__(self, clip_length):
        self.audio_files = glob.glob("datasets/speech/*.wav")
        random.Random(12321).shuffle(self.audio_files)  # make them appear in a random order, set seed for reproducibility
        # We have to do the above otherwise it is likely we train on one speaker for a bit, and then move on to another, etc, possibly not generalizing the model then
        self.clip_lenth = clip_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index): # TODO add cases when audio ends and we just have blank
        sound, sample_rate = torchaudio.load(self.audio_files[index])
        assert sound.shape[0] == 1
        assert sample_rate == 16000
        if sound.shape[1] > self.clip_lenth:
            start = random.randint(0, sound.shape[1] - self.clip_lenth - 1)
            return sound[:, start: start+self.clip_lenth]
        else:
            padding = torch.zeros(1, self.clip_lenth)
            padding[:, :sound.shape[1]] = sound
            return sound
        
#  Shamelessly copied from AcademiCodec norm.py
class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape, **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return x
        

class EncodeModel(nn.Module):
    def __init__(self, in_len, out_len):
        super().__init__()
        mod = []
        x = nn.Conv1d(1, 5, kernel_size=7, padding="same")
        mod += [x, nn.Sigmoid()]
        # for i in range(5):
        #     mod += [
        #         nn.Conv1d(1, 1, kernel_size=12, stride=6), nn.Sigmoid()
        #     ]

        mod += [
                nn.Linear(in_len * 5, 120), nn.Sigmoid()
            ]
        for j in range(2):
            mod += [
                nn.Linear(120, 120), nn.Sigmoid()
            ]
        
        self.model = nn.Sequential(*mod)

    def forward(self, x):
        return self.model(x)

class DecodeModel(nn.Module):
    def __init__(self, in_len, out_len):
        super().__init__()
        mod = []

        for j in range(2):
            mod += [
                nn.Linear(120, 120), nn.Sigmoid()
            ]

        
        mod += [
                nn.Linear(120, 1 * in_len * 5), nn.Sigmoid()
            ]

        # for i in range(5):
        #     mod += [
        #         nn.Conv1d(1, 1, kernel_size=3, stride=6), nn.Sigmoid()
        #     ]
            
        mod += [nn.Conv1d(5, 1, kernel_size=7, padding="same"), nn.Sigmoid()]
        self.model = nn.Sequential(*mod)

    def forward(self, x):
        return self.model(x)

class Full(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.encode = EncodeModel(length, -1)
        self.decode = DecodeModel(length, -1)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
