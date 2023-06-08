import sys, os
sys.path.append(os.getcwd() + '/testing') # a dodgy hack to improve

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
from tqdm import tqdm
writer = SummaryWriter(log_dir="logs/")

context_length = 240 * 5
batch_size = 64
TENSORBOARD_INTERAVAL = 25
VALID_SAVE_INTERVAL = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = TrainSpeechDataset(context_length)
valid_loader = ValidateSpeechDataset(48)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_loader, batch_size=batch_size)

encoder = models.Encoder(128).to(device)
quantizer = vq.RVQ(8, 1024, 128).to(device)
decoder = models.Decoder(128).to(device)

encoder.load_state_dict(torch.load("logs/encoder.state"))
decoder.load_state_dict(torch.load("logs/decoder.state"))
quantizer.load_state_dict(torch.load("logs/quantizer.state"))

encoder.eval()
decoder.eval()
quantizer.eval()
quantize_train = None
data = torch.zeros((8, 1024), dtype=torch.float32).to(device)
with torch.no_grad():
    for i, truth in enumerate(tqdm(train_dataloader)):
        truth = truth.to(device)
        outputs = encoder(truth)
        quantizer_loss = 0

        outputs = torch.transpose(outputs, 1, 2)  # BCT -> BTC

        _, indices, _ = quantizer(outputs)
        indices = torch.flatten(indices, end_dim=-2).t()
        o = torch.nn.functional.one_hot(indices, num_classes=1024)
        counts = torch.sum(o, dim=1)
        data = data + counts
        # indices = torch.flatten(indices).cpu().detach().numpy()

print(data.shape)
for i in range(8):
    for x in range(3):
        print(i, x, len((data[i] == x).nonzero(as_tuple=True)[0]))
    
    tk = torch.topk(data[i], 10)
    print(tk)
    print(torch.sum(data[i]).item())
    print(torch.sum(tk.values).item() / torch.sum(data[i]).item())