from model.datasets import *
import torch
from model import utils
import matplotlib.pyplot as plt
import itertools
from torchaudio import transforms
import torch.nn.functional as F
from torch import nn
from model import models
import torch
from model import vq
import argparse
import file_structure
from model.utils import norm

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = models.Encoder(256).to(device)
quantizer = vq.RVQ(8, 1024, 256).to(device)
decoder = models.Decoder(256).to(device)
encoder.load_state_dict(torch.load("logs/encoder.state"))
decoder.load_state_dict(torch.load("logs/decoder.state"))
quantizer.load_state_dict(torch.load("logs/quantizer.state"))

def codebooks_to_wav(indices):
    tensor = torch.tensor(indices).unsqueeze(0).to("cuda")
    x = quantizer.decode(tensor)
    x = torch.transpose(x, 1, 2)
    wav_data = decoder(x).squeeze(0)
    torchaudio.save(f"test.wav", wav_data.to("cpu"), sample_rate=16000)


def wav_to_codebooks(audio_data):
    audio_data = audio_data.unsqueeze(0).to(device)
    x = encoder(audio_data)
    x = torch.transpose(x, 1, 2)
    values = quantizer.encode(x)
    return values


def save_to_file(codebooks):
    f = file_structure.File(codebooks.squeeze(0).to("cpu"), data_bit_depth=10, n_codebooks=5)
    f.write("saved.sc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compresser")
    parser.add_argument("filename")
    args = parser.parse_args()

    if args.filename.endswith("sc"):
        f = file_structure.File.read(args.filename)
        codebooks_to_wav(f.data)

    else:
        sound, sample_rate = torchaudio.load(args.filename)
        sound = norm(sound)
        torchaudio.save(f"x.wav", sound, sample_rate=16000)
        assert sample_rate == 16000
        padding = torch.zeros(1, 240 * math.ceil(sound.shape[1] / 240))
        padding[:, :sound.shape[1]] = sound
        save_to_file(wav_to_codebooks(padding))
