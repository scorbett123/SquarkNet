import torch
import torchaudio
from torch.utils.data import Dataset
import glob
import random
import matplotlib.pyplot as plt
import math
from model import utils
from torchaudio.functional import resample

def make_length(audio, length):
        if audio.shape[1] > length:
            start = random.randint(0, audio.shape[1] - length - 1)
            return audio[:, start: start+length]
        else:
            padding = torch.zeros(1, length)
            padding[:, :audio.shape[1]] = audio
            return padding

class LibriTTS(Dataset):
    def __init__(self, clip_length, length=None):
        self._audio_files = glob.glob("datasets/speech_train/*.wav")
        random.Random(12321).shuffle(self._audio_files)  # make them appear in a random order, set seed for reproducibility
        if length != None:
            self._audio_files = self._audio_files[:length]
        # We have to do the above otherwise it is likely we train on one speaker for a bit, and then move on to another, etc, possibly not generalizing the model then
        self._clip_length = clip_length

    def __len__(self):
        return len(self._audio_files)

    def __getitem__(self, index): # TODO add cases when audio ends and we just have blank
        sound, sample_rate = torchaudio.load(self._audio_files[index % len(self._audio_files)])
        assert sound.shape[0] == 1, "Only mono audio allowed, no stereo"
        assert sample_rate == 16000, "Sample rate of file isn't 16 kHz"
        sound = utils.norm(sound)  # Want to make sure that we normalize over the whole clip, if we only normalize over our sample, silences may just become noise
        return make_length(sound, self._clip_length)
        
        
class CommonVoice(Dataset):
    def __init__(self, clip_length, mode: str="train", path: str="/mnt/d/datasets/CommonVoice/cv-corpus-14.0-2023-06-23/en") -> None:
        super().__init__()
        self._path = path
        self._clip_length = clip_length
        
        with open(f"{path}/{mode}.tsv") as f:
            lines = f.readlines()

        titles = lines[0].split("\t")
        self._data = [{titles[j]: d for j, d in enumerate(line.split("\t"))} for line in lines[1:]]
        
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index) -> torch.Tensor:
        sound, sample_rate = torchaudio.load(f"{self._path}/clips/{self._data[index]['path']}")
        if sample_rate != 160000:
            sound = resample(sound, sample_rate, 16000)
        sound = utils.norm(sound)
        
        return make_length(sound, self._clip_length)
        

class RandomAudioDataset(Dataset):
    def __init__(self, clip_length, length):
        self._clip_length = clip_length
        self._length = length

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return torch.randn((1, self._clip_length))
    

class SetAudioDataset(Dataset):
    def __init__(self, clip_length, length, value):
        self._clip_length = clip_length
        self._length = length
        self._value = value.unsqueeze(0)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._value
        

class ValidateSpeechDataset(Dataset):
    def __init__(self, clip_length):
        """ all will be multiples of clip_length """
        self._audio_files = glob.glob("datasets/speech_valid/*.wav")
        random.Random(12321).shuffle(self._audio_files)  # make them appear in a random order, set seed for reproducibility
        # We have to do the above otherwise it is likely we train on one speaker for a bit, and then move on to another, etc, possibly not generalizing the model then
        self._audio_files = self._audio_files[:5]
        self._clip_length = clip_length

    def __len__(self):
        return len(self._audio_files)

    def __getitem__(self, index): # TODO add cases when audio ends and we just have blank
        sound, sample_rate = torchaudio.load(self._audio_files[index % len(self._audio_files)])
        sound = utils.norm(sound)
        assert sound.shape[0] == 1
        assert sample_rate == 16000

        #padding = torch.zeros(1, self.clip_lenth * math.ceil(sound.shape[1] / self.clip_lenth))
        padding = torch.zeros(1, 16000 * 30)
        padding[:, :sound.shape[1]] = sound
        return padding



if __name__ == "__main__":
    cv = CommonVoice(4000)
    print(cv[0].shape)