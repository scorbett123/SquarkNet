from scipy.io import wavfile
from pesq import pesq
import glob
import os
from tqdm import tqdm
import random
from encodec.utils import convert_audio
import torch
from model.models import Models
import torchaudio
import math
import torch

model = Models.load("model_saves/low_qual.saved", device="cuda")
model._load_from_state_dict

audio_files = random.sample(glob.glob("datasets/speech_train/*.wav"), 500)
values = []

with torch.no_grad():
    for file in tqdm(audio_files):
        ref, rate = torchaudio.load(file)
        ref = ref.to("cuda")

        res = model.forward(ref)[0]

        torchaudio.save("evaluation/tmp/test.wav", res.cpu(), rate)
        values.append(pesq(rate, ref.squeeze(0).cpu().numpy(), res.squeeze(0).cpu().numpy(), 'nb'))

        # os.remove(f"evaluation/tmp/{file.split('.')[0].split('/')[-1]}.lyra")
        # os.remove(f"evaluation/tmp/{file.split('.')[0].split('/')[-1]}_decoded.wav")

    print(sum(values) / len(values))