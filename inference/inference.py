from model.datasets import *
import torch
from model import models
import torch
import file_structure
from model.utils import norm
import torchaudio

def sc_to_wav(path, output_path, model: models.Models):
    if not output_path.endswith(".wav"):
        output_path += ".wav"

    f = file_structure.File.read(path)

    indices = torch.tensor(f.data).unsqueeze(0)

    out = model.decode(indices)
    torchaudio.save(output_path, out.cpu(), sample_rate=16000)
    print("done")


def wav_to_sc(path, output_path, model: models.Models):
    if not output_path.endswith(".sc"):
        output_path += ".sc"
    sound, sample_rate = torchaudio.load(path)
    sound = norm(sound)

    if sample_rate != 16000:
        sound = torchaudio.functional.resample(sound, orig_freq=sample_rate, new_freq=16000)

    audio_data = sound.unsqueeze(0)
    codebooks = model.encode(audio_data)

    f = file_structure.File(codebooks.squeeze(0), data_bit_depth=math.ceil(math.log2(model.ncodes)), n_codebooks=model.nbooks)
    f.write(output_path)
    print("done")

