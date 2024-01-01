from pesq import pesq
import glob
from tqdm import tqdm
import random
from encodec.utils import convert_audio
import torch
from model.models import Models
import torchaudio
import math
import torch

from model.utils import norm

model = Models.load("model_saves/highqual.saved", device="cuda")

audio_files = random.sample(glob.glob("datasets/speech_train/*.wav"), 500)
values = []

with torch.no_grad():
    for file in tqdm(audio_files):  # inefficient to not batch, but this is only iterating over 500, it's fine
        ref, rate = torchaudio.load(file)
        ref = ref.to("cuda")
        ref = norm(ref)
        res = model.forward(ref)[0]

        torchaudio.save("evaluation/tmp/test.wav", res.cpu(), rate)
        values.append(pesq(rate, ref.squeeze(0).cpu().numpy(), res.squeeze(0).cpu().numpy(), 'nb'))

    print(sum(values) / len(values))