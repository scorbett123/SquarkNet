from stuff import *
import torch
import utils
import matplotlib.pyplot as plt
import itertools
from torchaudio import transforms
import torch.nn.functional as F
from torch import nn
import models

context_length = 240 * 40
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = TrainSpeechDataset(context_length)
valid_loader = ValidateSpeechDataset(48)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_loader, batch_size=1)

error = nn.L1Loss()

encoder = models.Encoder(256).to(device)
decoder = models.Decoder(256).to(device)

spec = transforms.MelSpectrogram(16000, n_mels=80, n_fft=1024, hop_length=240, win_length=1024).to(device)

optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.0002, betas=[0.5, 0.9])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
losses = []
x = 0
av = 0

# print(encoder)
# print(decoder)
e = 0
while True:
    e+= 1
    encoder.train()
    for truth in train_dataloader:
        truth = truth.to(device)
        outputs = encoder(truth)
        predicted_in = decoder(outputs)

        loss = F.l1_loss(spec(predicted_in), spec(truth.detach()))
        av += loss.item()
        x += 1
        if x % 50 == 0:
            losses.append(av/50)
            av = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH, DO NOT FORGET THIS, I spent a long time wondering "why isn't it learning anything"
    print(e, sum(losses[-60:]) / (min(60, max(len(losses), 1))))
    if e % 10 == 0:
        for i, j in enumerate(valid_loader):
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                original = j.to(device)
                result = encoder(original)
                result = decoder(result)

                utils.plot_spectrograms(spec(original[0]), spec(result[0]), file=f"{i}-truth.png")
                torchaudio.save(f"{i}-encoded.wav", result[0].to("cpu"), sample_rate=16000)
                torchaudio.save(f"{i}-clean.wav", j[0], sample_rate=16000)
                plt.close()
                plt.clf()
    

        ax = plt.subplot()
        ax.plot([i for i in range(len(losses))], losses )
        plt.savefig("matplotlib.png")
