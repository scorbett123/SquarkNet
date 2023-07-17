import torch
from torch.nn.utils import weight_norm
from torch import nn
import torchaudio
from model import datasets
from torch.utils.data import DataLoader
import math
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="logs_test/")
LEAKY_RELU = 0.2
INIT_MEAN = 0.0
INIT_STD = 0.01

def get_padding_nd(kernel_sizes, dilations):
    return [math.floor((kernel_size-1) * dilation / 2) for kernel_size, dilation in zip(kernel_sizes, dilations)]


class STFTDiscriminator(torch.nn.Module):
    def __init__(self,
                 kernel_size=(3,9)) -> None:
        super().__init__()
        self.transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256, win_length=1024, normalized=True, power=None, center=False, pad_mode=None)
        self.conv1 = weight_norm(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=get_padding_nd((3,9), (1,1))))
        self.conv2 = weight_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=(1,2), dilation=(1,1), padding=get_padding_nd((3,9), (1,1))))
        self.conv3 = weight_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=(1,2), dilation=(2,1), padding=get_padding_nd((3,9), (2,1))))
        self.conv4 = weight_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=(1,2), dilation=(4,1), padding=get_padding_nd((3,9), (4,1))))
        self.conv5 = weight_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=get_padding_nd((3,3), (1,1))))
        self.conv6 = weight_norm(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=get_padding_nd((3,3), (1,1))))

        nn.init.normal_(self.conv1.weight, INIT_MEAN, INIT_STD)
        nn.init.normal_(self.conv2.weight, INIT_MEAN, INIT_STD)
        nn.init.normal_(self.conv3.weight, INIT_MEAN, INIT_STD)
        nn.init.normal_(self.conv4.weight, INIT_MEAN, INIT_STD)
        nn.init.normal_(self.conv5.weight, INIT_MEAN, INIT_STD)
        nn.init.normal_(self.conv6.weight, INIT_MEAN, INIT_STD)


    def forward(self, x):
        spec: torch.Tensor = self.transform(x)
        spec = torch.cat([spec.real, spec.imag], dim=-1)
        spec = spec.transpose(-1, -2)
        spec = self.conv1(spec)
        spec = torch.nn.functional.leaky_relu(spec, LEAKY_RELU)
        spec = self.conv2(spec)
        spec = torch.nn.functional.leaky_relu(spec, LEAKY_RELU)

        spec = self.conv3(spec)
        spec = torch.nn.functional.leaky_relu(spec, LEAKY_RELU)

        spec = self.conv4(spec)
        spec = torch.nn.functional.leaky_relu(spec, LEAKY_RELU)

        spec = self.conv5(spec)
        spec = torch.nn.functional.leaky_relu(spec, LEAKY_RELU)
        logits = self.conv6(spec)
        spec = torch.nn.functional.tanh(spec)

        return logits.view(logits.shape[0], -1).mean(dim=1) # dunno if this is meant to be here, not mentioned in paper, but makes sense to me... TODO when there is bugs this is probably it
    


if __name__ == "__main__":
    loader = DataLoader(datasets.TrainSpeechDataset(240*48), 20)
    random_loader = DataLoader(datasets.RandomAudioDataset(240*48, 100), 20)
    discrim = STFTDiscriminator()
    optim = torch.optim.Adam(discrim.parameters(),  lr=0.002, betas=[0.5, 0.9])
    discrim.train()
    while True:
        for actual, random in zip(loader, random_loader):
            print(x:=discrim(actual))
            print(y:=discrim(random))

            a = 1-x  # we want x to be high
            b = 1+y  # we want y to be low
            a[a<0] = 0  # if x already above 1 we want to ignore in the loss
            b[b<0] = 0  # if y already below -1 we want to ignore in the loss
            loss = torch.mean(a)
            loss += torch.mean(b)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(loss)
            writer.add_scalar("loss", loss.item())

