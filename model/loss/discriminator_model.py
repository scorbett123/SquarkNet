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

def get_padding_nd(kernel_sizes: list[int], dilations: list[int]) -> list[int]:
    return [math.floor((kernel_size-1) * dilation / 2) for kernel_size, dilation in zip(kernel_sizes, dilations)]


class STFTDiscriminator(torch.nn.Module):
    """ Written as in Encodec paper, slightly better than soundstream implementation I think, TODO try both and see which is better"""
    def __init__(self,
                 scale,
                 kernel_size=(3,8)) -> None:
        super().__init__()
        self._transform = torchaudio.transforms.Spectrogram(n_fft=(scale - 1) * 2, win_length=scale, normalized=False, power=None, center=False, pad_mode=None)

        self._convs = nn.ModuleList([
            weight_norm(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=kernel_size, padding=get_padding_nd((3,8), (1,1)))),
            weight_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=(1,2), dilation=(1,1), padding=get_padding_nd((3,8), (1,1)))),
            weight_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=(1,2), dilation=(2,1), padding=get_padding_nd((3,8), (2,1)))),
            weight_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=kernel_size, stride=(1,2), dilation=(4,1), padding=get_padding_nd((3,8), (4,1)))),
            weight_norm(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=get_padding_nd((3,3), (1,1)))),
            weight_norm(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=get_padding_nd((3,3), (1,1))))
        ])

        for conv in self._convs:
            nn.init.normal_(conv.weight, INIT_MEAN, INIT_STD)


    def forward(self, x):
        spec: torch.Tensor = self._transform(x)
        spec = torch.cat([spec.real, spec.imag], dim=-1)
        spec = spec.transpose(-1, -2)
        internal_activations = []  # will be in form L B X Y

        for conv in self._convs:
            spec = conv(spec)
            spec = torch.nn.functional.leaky_relu(spec, LEAKY_RELU)
            internal_activations.append(spec)
        
        logits = torch.nn.functional.tanh(spec)

        return logits.view(logits.shape[0], -1).mean(dim=1), internal_activations # dunno if this is meant to be here, not mentioned in paper, but makes sense to me... TODO when there is bugs this is probably it
    
class MultiScaleSTFTDiscriminator(torch.nn.Module):
    def __init__(self, scales = [2048, 1024, 512, 256, 128]) -> None:
        super().__init__()
        self._discrims = torch.nn.ModuleList([STFTDiscriminator(scale) for scale in scales])

    def forward(self, x):
        logit, disc_feature = self._discrims[0](x)
        result = logit.unsqueeze(0).transpose(0, 1)

        features = [disc_feature]  # need to use lists :( due to differing lengths of dimensions, too hard to exclude at the other end

        for discrim in self._discrims[1:]:
            logit, disc_feature = discrim(x)
            result = torch.concat((logit.unsqueeze(0).transpose(0, 1), result), dim=1)
            features.append(disc_feature)
        return result, features  # D B L X Y to B D L X Y, keep batch first (convention)

if __name__ == "__main__":
    loader = DataLoader(datasets.LibriTTS(240*48), 20)
    random_loader = DataLoader(datasets.RandomAudioDataset(240*48, 100), 20)
    discrim = MultiScaleSTFTDiscriminator()
    optim = torch.optim.Adam(discrim.parameters(),  lr=0.002, betas=[0.5, 0.9])
    discrim.train()
    while True:
        for actual, random in zip(loader, random_loader):
            x = discrim(actual)
            y = discrim(random)

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

