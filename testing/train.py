from stuff import *
import torch
import utils
import matplotlib.pyplot as plt
import itertools
from torchaudio import transforms
import torch.nn.functional as F
from torch import nn
import models
import torch
import vq
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="logs/")

context_length = 48 * 200
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = TrainSpeechDataset(context_length)
valid_loader = ValidateSpeechDataset(48)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_loader, batch_size=1)

error = nn.L1Loss()

encoder = models.Encoder(256).to(device)
quantizer = vq.RVQ(5, 1024, 256).to(device)
decoder = models.Decoder(256).to(device)

spec = transforms.MelSpectrogram(16000, n_mels=80, n_fft=1024, hop_length=240, win_length=1024, f_max=8000, f_min=0).to(device)
# encoder.load_state_dict(torch.load("logs/encoder.state"))
# decoder.load_state_dict(torch.load("logs/decoder.state"))
encoder.load_state_dict(torch.load("logs/encoder.state"))
decoder.load_state_dict(torch.load("logs/decoder.state"))
quantizer.load_state_dict(torch.load("logs/quantizer.state"))


optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters(), quantizer.parameters()), lr=0.0002, betas=[0.5, 0.9])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
losses = []
x = 0
av = 0
torch.autograd.set_detect_anomaly(True)
# print(encoder)
# print(decoder)
# quantizer_enable_epoch = 2
e = 0
while True:
    e+= 1
    encoder.train()
    decoder.train()
    quantizer.train()
    quantize_train = None
    for truth in train_dataloader:
        truth = truth.to(device)
        outputs = encoder(truth)
        #print(outputs.shape)
        # quantizer_loss = 0
        # if e > quantizer_enable_epoch:
        #     outputs, quantizer_loss = quantizer(outputs)
        # elif e == quantizer_enable_epoch:
        #     if quantize_train != None:
        #         quantize_train = torch.cat((quantize_train, outputs.detach()), dim=0)
        #     else:
        #         quantize_train = torch.clone(outputs.detach())
        #     print("adding")
        #     if quantize_train.shape[0] > 300:
        #         break
        #     continue
            
        outputs = torch.transpose(outputs, 1, 2)  # BCT -> BTC
        #print(outputs.shape)
        outputs, quantizer_loss = quantizer(outputs)
        outputs = torch.transpose(outputs, 2,1)  # BTC -> BCT
        
        predicted_in = decoder(outputs)

        loss = F.l1_loss(spec(predicted_in), spec(truth)) + quantizer_loss
        av += loss.item()
        x += 1
        if x % 10 == 0:
            losses.append(av/10)
            av = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH, DO NOT FORGET THIS, I spent a long time wondering "why isn't it learning anything"
    if len(losses) > 0:
        print(e, losses[-1])
        writer.add_scalar("loss", losses[-1], e)
        writer.flush()
    # if e == quantizer_enable_epoch:
    #     print(quantize_train.shape)
    #     quantizer.initialise(quantize_train)
    if e % 3 == 0:
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
    
        torch.save(encoder.state_dict(), "logs/encoder.state")
        torch.save(decoder.state_dict(), "logs/decoder.state")
        torch.save(quantizer.state_dict(), "logs/quantizer.state")

        ax = plt.subplot()
        ax.plot([i for i in range(len(losses))], losses )
        plt.savefig("matplotlib.png")
