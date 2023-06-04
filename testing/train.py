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
import whispertesting
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="logs/")

context_length = 48 * 200
batch_size = 64
TENSORBOARD_INTERAVAL = 25
VALID_SAVE_INTERVAL = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = TrainSpeechDataset(context_length)
valid_loader = ValidateSpeechDataset(48)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_loader, batch_size=batch_size)

error = nn.L1Loss()

encoder = models.Encoder(256).to(device)
quantizer = vq.RVQ(16, 512, 256).to(device)
decoder = models.Decoder(256).to(device)

custommel = whispertesting.CustomMel().to(device)
spec = transforms.MelSpectrogram(16000, n_mels=80, n_fft=1024, hop_length=240, f_max=8000, f_min=0).to(device)
whisper = whispertesting.WhisperLoss(context_length, batch_size)

# encoder.load_state_dict(torch.load("logs/encoder.state"))
# decoder.load_state_dict(torch.load("logs/decoder.state"))
# encoder.load_state_dict(torch.load("logs/encoder.state"))
# decoder.load_state_dict(torch.load("logs/decoder.state"))
# quantizer.load_state_dict(torch.load("logs/quantizer.state"))


optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters(), quantizer.parameters()), lr=0.0002, betas=[0.5, 0.9])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
losses = {"loss" : [], "spec1": [], "spec2": [], "whisper": [], "quantization": []}


def calc_loss(predicted_in, truth, quantizer_loss, eval=False):
    spec_1p = spec(predicted_in)
    spec_1t = spec(truth)
    loss_spec_1 =  F.mse_loss(spec_1p, spec_1t) / (750 * 4)
    if not eval:
        losses["spec1"].append(loss_spec_1.item())

    if not eval:
        whisper_batched_p = whisper.process_batch(predicted_in)
        whisper_batched_t = whisper.process_batch(truth)
    else:
        whisper_batched_p = predicted_in.squeeze(1)
        whisper_batched_t = truth.squeeze(1)
    spec_2p = custommel(whisper_batched_p)
    spec_2t = custommel(whisper_batched_t)
    loss_spec_2 = F.mse_loss(spec_2p, spec_2t) * 20
    if not eval:
        losses["spec2"].append(loss_spec_2.item())

    whisper_p = whisper(spec_2p)
    whisper_t = whisper(spec_2t)
    loss_whisper = F.mse_loss(whisper_p, whisper_t)
    if not eval:
        losses["whisper"].append(loss_whisper.item())

    loss = loss_spec_1  + quantizer_loss + loss_spec_2 + loss_whisper
    if not eval:
        losses["loss"].append(loss.item())
    return loss
# print(encoder)
# print(decoder)
# quantizer_enable_epoch = 200
e = 0
steps = 0
while e:=e+1:  # sligtly dodgy, however why not? I'm just messing around a bit
    encoder.train()
    decoder.train()
    quantizer.train()
    quantize_train = None
    for truth in train_dataloader:
        truth = truth.to(device)
        outputs = encoder(truth)
        quantizer_loss = 0
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
        outputs, _, quantizer_loss = quantizer(outputs)
        outputs = torch.transpose(outputs, 2,1)  # BTC -> BCT
        
        predicted_in = decoder(outputs)

        losses["quantization"].append(quantizer_loss.item())
        loss = calc_loss(predicted_in, truth, quantizer_loss)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH, DO NOT FORGET THIS, I spent a long time wondering "why isn't it learning anything"

        if (steps:=(steps+1)) % TENSORBOARD_INTERAVAL == 0:
            print(f"{steps} steps done")
            for i in losses:
                writer.add_scalar(f"loss/{i}", sum(losses[i][-TENSORBOARD_INTERAVAL: ])/ min(len(losses["loss"]), TENSORBOARD_INTERAVAL), steps)
            writer.flush()
        if steps % VALID_SAVE_INTERVAL == 0:
            encoder.eval()
            decoder.eval()
            
            lossx = []
            for i, j in enumerate(valid_loader):
                with torch.no_grad():
                    original = j.to(device)
                    result = encoder(original)
                    result, _, quantizer_loss = quantizer(result.transpose(-1, -2))
                    result = decoder(result.transpose(-1, -2))

                    l = calc_loss(result, original, quantizer_loss, eval=True)
                    lossx.append(l / j.shape[0])

                    if i == 0:
                        for x in range(4):
                            utils.plot_spectrograms(spec(original[x, :16000]), spec(result[x, :16000]), file=f"{x}-truth.png")
                            torchaudio.save(f"{x}-encoded.wav", result[x].to("cpu"), sample_rate=16000)
                            torchaudio.save(f"{x}-clean.wav", j[x], sample_rate=16000)
                            plt.close()
                            plt.clf()
            
            writer.add_scalar(f"validloss/{i}", sum(lossx) / len(lossx), steps)
            encoder.train()
            decoder.train()
            quantizer.train()
        
            torch.save(encoder.state_dict(), "logs/encoder.state")
            torch.save(decoder.state_dict(), "logs/decoder.state")
            torch.save(quantizer.state_dict(), "logs/quantizer.state")
        
    scheduler.step()
    print(f"EPOCH {e} DONE!!!!")
    
    # if e == quantizer_enable_epoch:
    #     print(quantize_train.shape)
    #     quantizer.initialise(quantize_train)
        