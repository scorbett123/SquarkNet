from stuff import *
import torch
import utils
import matplotlib.pyplot as plt
import itertools
from torchaudio import transforms
import torch.nn.functional as F
from torch import nn
import models

context_length = 12000
device = "cuda" if torch.cuda.is_available() else "cpu"

data = SpeechDataset(context_length)
train_dataloader = DataLoader(data, batch_size=64, shuffle=True)

error = nn.L1Loss()

encoder = models.Encoder(256).to(device)
decoder = models.Decoder(256).to(device)



spec = transforms.MelSpectrogram(16000).to(device)

optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.0002, betas=[0.5, 0.9])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
losses = []
x = 0
av = 0

# print(encoder)
# print(decoder)
for e in range(140):
    encoder.train()
    for truth in train_dataloader:
        truth = truth.to(device)
        outputs = encoder(truth)
        predicted_in = decoder(outputs)

        loss = F.l1_loss(spec(predicted_in), spec(truth.detach()))
        av += loss.item()
        x += 1
        if x % 10 == 0:
            losses.append(av/10)
            av = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH, DO NOT FORGET THIS, I spent a long time wondering "why aren't we learning anything"
    print(e, sum(losses[-30:]) / (min(30, len(losses))))
    if e % 10 == 0:
        for i in range(3):
            test = data[i].unsqueeze(0).to("cuda")
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                result = encoder(test)
                result = decoder(result)
                torchaudio.save(f"{i}-encoded.wav", result[0].to("cpu"), sample_rate=16000)
                torchaudio.save(f"{i}-clean.wav", test[0].to("cpu"), sample_rate=16000)
                utils.plot_spectrograms(spec(test[0]), spec(result[0]), file=f"{i}-truth.png")
                plt.close()
                plt.clf()

ax = plt.subplot()
ax.plot([i for i in range(len(losses))], losses )
plt.savefig("matplotlib.png")
