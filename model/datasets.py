import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import glob
import random
import matplotlib.pyplot as plt
import math
from model import utils

class TrainSpeechDataset(Dataset):
    def __init__(self, clip_length):
        self.audio_files = glob.glob("datasets/speech_train/*.wav")
        random.Random(12321).shuffle(self.audio_files)  # make them appear in a random order, set seed for reproducibility
        # We have to do the above otherwise it is likely we train on one speaker for a bit, and then move on to another, etc, possibly not generalizing the model then
        self.clip_lenth = clip_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index): # TODO add cases when audio ends and we just have blank
        sound, sample_rate = torchaudio.load(self.audio_files[index % len(self.audio_files)])
        assert sound.shape[0] == 1, "Only mono audio allowed, no stereo"
        assert sample_rate == 16000, "Sample rate of file isn't 16 kHz"
        sound = utils.norm(sound)  # Want to make sure that we normalize over the whole clip, if we only normalize over our sample, silences may just become noise
        if sound.shape[1] > self.clip_lenth:
            start = random.randint(0, sound.shape[1] - self.clip_lenth - 1)
            return sound[:, start: start+self.clip_lenth]
        else:
            padding = torch.zeros(1, self.clip_lenth)
            padding[:, :sound.shape[1]] = sound
            return sound
        
class RandomAudioDataset(Dataset):
    def __init__(self, clip_length, length):
        self.clip_length = clip_length
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.randn((1, self.clip_length))
    

class SetAudioDataset(Dataset):
    def __init__(self, clip_length, length, value):
        self.clip_length = clip_length
        self.length = length
        self.value = value.unsqueeze(0)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.value
        

class ValidateSpeechDataset(Dataset):
    def __init__(self, clip_length):
        """ all will be multiples of clip_length """
        self.audio_files = glob.glob("datasets/speech_valid/*.wav")
        random.Random(12321).shuffle(self.audio_files)  # make them appear in a random order, set seed for reproducibility
        # We have to do the above otherwise it is likely we train on one speaker for a bit, and then move on to another, etc, possibly not generalizing the model then
        self.audio_files = self.audio_files[:5]
        self.clip_lenth = clip_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index): # TODO add cases when audio ends and we just have blank
        sound, sample_rate = torchaudio.load(self.audio_files[index % len(self.audio_files)])
        sound = utils.norm(sound)
        assert sound.shape[0] == 1
        assert sample_rate == 16000

        #padding = torch.zeros(1, self.clip_lenth * math.ceil(sound.shape[1] / self.clip_lenth))
        padding = torch.zeros(1, 16000 * 30)
        padding[:, :sound.shape[1]] = sound
        return padding
