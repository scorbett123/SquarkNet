import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import glob
import random
import matplotlib.pyplot as plt
import math

class TrainSpeechDataset(Dataset):
    def __init__(self, clip_length):
        self.audio_files = glob.glob("datasets/speech/*.wav")
        random.Random(12321).shuffle(self.audio_files)  # make them appear in a random order, set seed for reproducibility
        # We have to do the above otherwise it is likely we train on one speaker for a bit, and then move on to another, etc, possibly not generalizing the model then
        self.clip_lenth = clip_length

    def __len__(self):
        return len(self.audio_files)*10

    def __getitem__(self, index): # TODO add cases when audio ends and we just have blank
        sound, sample_rate = torchaudio.load(self.audio_files[index % len(self.audio_files)])
        assert sound.shape[0] == 1
        assert sample_rate == 16000
        if sound.shape[1] > self.clip_lenth:
            start = random.randint(0, sound.shape[1] - self.clip_lenth - 1)
            return sound[:, start: start+self.clip_lenth]
        else:
            padding = torch.zeros(1, self.clip_lenth)
            padding[:, :sound.shape[1]] = sound
            return sound
        

class ValidateSpeechDataset(Dataset):
    def __init__(self, clip_length):
        """ all will be multiples of clip_length """
        self.audio_files = glob.glob("datasets/speech/*.wav")
        random.Random(12321).shuffle(self.audio_files)  # make them appear in a random order, set seed for reproducibility
        # We have to do the above otherwise it is likely we train on one speaker for a bit, and then move on to another, etc, possibly not generalizing the model then
        self.audio_files = self.audio_files[:5]
        self.clip_lenth = clip_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index): # TODO add cases when audio ends and we just have blank
        sound, sample_rate = torchaudio.load(self.audio_files[index % len(self.audio_files)])
        assert sound.shape[0] == 1
        assert sample_rate == 16000

        padding = torch.zeros(1, self.clip_lenth * math.ceil(sound.shape[1] / self.clip_lenth))
        padding[:, :sound.shape[1]] = sound
        return sound
