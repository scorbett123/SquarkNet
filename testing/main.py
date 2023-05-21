from stuff import *
import torch
import utils
import matplotlib.pyplot as plt
import itertools
from torchaudio import transforms
import torch.nn.functional as F
from torch import nn
import models

context_length = 320

data = SpeechDataset(context_length)
train_dataloader = DataLoader(data, batch_size=64, shuffle=True)

error = nn.L1Loss()

encoder = models.Encoder(context_length, 128).to("cuda")
#decoder = hificodec.Generator(None).to("cuda")



spec = transforms.MelSpectrogram(16000).to("cuda")

optimizer = torch.optim.Adam(itertools.chain(encoder.parameters()), lr=0.0002, betas=[0.5, 0.9])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
losses = []
x = 0
av = 0


for e in range(140):
    encoder.train()
    for truth in train_dataloader:
        truth = truth.to("cuda")
        outputs = encoder(torch.clone(truth))
        assert 1==0
        # loss = F.l1_loss(spec(outputs), spec(truth.detach()))
        # av += loss.item()
        # x += 1
        # if x % 10 == 0:
        #     losses.append(av/10)
        #     av = 0

        # loss.backward()
    # print(e, sum(losses[-30:]) / (min(30, len(losses))))
    # if e % 5 == 0:
    #     for i in range(3):
    #         test = data[i].unsqueeze(0).to("cuda")
    #         print(test.shape)
    #         encoder.eval()
    #         decoder.eval()
    #         with torch.no_grad():
    #             result = encoder(test)
    #             result = decoder(result)
    #             utils.plot_spectrograms(spec(test[0]), spec(result[0]), file=f"{i}-truth.png")
    #             plt.close()
    #             plt.clf()

ax = plt.subplot()
ax.plot([i for i in range(len(losses))], losses )
plt.savefig("matplotlib.png")
