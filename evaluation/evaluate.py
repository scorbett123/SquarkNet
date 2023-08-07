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

model = Models(256, 8, 1024, device="cuda", discrim=False)
model.load("logs-t")

audio_files = random.sample(glob.glob("datasets/speech_train/*.wav"), 500)
values = []

for file in tqdm(audio_files):
    ref, rate = torchaudio.load(file)
    ref = ref.to("cuda")

    padding_amount = 240 - round(((ref.shape[1] / 240) % 1) * 240)
    # print((ref.shape[1] / 240) % 1, round(((ref.shape[1] / 240) % 1) * 240))
    # print(padding_amount, ref.shape[1])
    ref = torch.nn.functional.pad(ref, [0, padding_amount])
    ref = ref.unsqueeze(0)
    with torch.no_grad():
        res, _ = model(ref)

    # print(res.shape, ref.shape)
    assert res.shape == ref.shape
    torchaudio.save("evaluation/tmp/test.wav", res.squeeze(0).cpu(), rate)
    values.append(pesq(rate, ref.squeeze(0).squeeze(0).cpu().numpy(), res.squeeze(0).squeeze(0).cpu().numpy(), 'wb'))

    # os.remove(f"evaluation/tmp/{file.split('.')[0].split('/')[-1]}.lyra")
    # os.remove(f"evaluation/tmp/{file.split('.')[0].split('/')[-1]}_decoded.wav")

print(sum(values) / len(values))