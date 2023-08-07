from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from model.loss.discriminator_model import MultiScaleSTFTDiscriminator
from model import vq
import math
import torch
LEAKY_RELU = 0.2
INIT_MEAN = 0.0
INIT_STD = 0.01


class Models(nn.Module):
    def __init__(self, n_channels, nbooks, ncodes, sftf_scales=[1024, 512, 256], device="cpu", discrim=True) -> None:  # improve type hints here, probably make a separate quantizer class
        super().__init__()

        self.encoder = Encoder(n_channels).to(device)
        self.quantizer = vq.RVQ(nbooks, ncodes, n_channels).to(device)
        self.decoder = Decoder(n_channels).to(device)
        if discrim:
            self.discriminator = MultiScaleSTFTDiscriminator(scales=sftf_scales).to(device)
        else:
            self.discriminator = None

    def forward(self, x):
        """ return in the form y, q_loss """
        encoded  = self.encoder(x)

        encoded = torch.transpose(encoded, 1, 2)
        after_q, _, q_loss  = self.quantizer(encoded)
        after_q = torch.transpose(after_q, 1, 2)
        y = self.decoder(after_q)
        return y, q_loss
    
    def discrim_forward(self, x):
        return self.discriminator(x)
    
    def load(self, path="logs-t"):
        self.encoder.load_state_dict(torch.load(f"{path}/encoder.state"))
        self.decoder.load_state_dict(torch.load(f"{path}/decoder.state"))
        self.quantizer.load_state_dict(torch.load(f"{path}/quantizer.state"))
        if self.discriminator != None:  
            self.discriminator.load_state_dict(torch.load(f"{path}/discriminator.state"))


def get_padding(kernel_size, dilation=1):
    return math.floor((kernel_size-1) * dilation / 2)

# paper https://arxiv.org/pdf/2009.02095.pdf
class ResidualUnit(nn.Module):
    """ Residual Unit, input and output are same dimension """
    def __init__(self, nChannels, dilation) -> None:
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(nChannels, nChannels, kernel_size=3, padding=get_padding(3, dilation), dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(nChannels, nChannels, kernel_size=1))
        nn.init.normal_(self.conv1.weight, INIT_MEAN, INIT_STD)
        nn.init.normal_(self.conv2.weight, INIT_MEAN, INIT_STD)

    def forward(self, x):
        xt = F.leaky_relu(x, LEAKY_RELU)
        xt = self.conv1(xt)
        xt = F.leaky_relu(xt, LEAKY_RELU)
        xt = self.conv2(xt)
        return xt + x 

class EncoderBlock(nn.Module):
    """ Encoder block, requires input size to be nOutChannels / 2 """
    def __init__(self, nOutChannels, stride) -> None:
        super().__init__()
        self.res_units = nn.ModuleList([
            ResidualUnit(int(nOutChannels/2), dilation=1), # in N out N
            ResidualUnit(int(nOutChannels/2), dilation=3), # in N out N
            ResidualUnit(int(nOutChannels/2), dilation=9) # in N out N
        ])
        self.conv = weight_norm(nn.Conv1d(int(nOutChannels/2), nOutChannels, kernel_size=2*stride, stride=stride, padding=get_padding(2*stride)))  # in N out 2N

    def forward(self, x):
        for unit in self.res_units:
            x = unit(x)
        
        x = self.conv(x)
        x = F.leaky_relu(x, LEAKY_RELU)
        return x
    
class DecoderBlock(nn.Module): # TODO for encoder + decoder check that padding isn't rediculous
    """ Encoder block, requires input size to be nOutChannels / 2 """
    def __init__(self, nOutChannels, stride, kernel) -> None:
        super().__init__()
        self.res_units = nn.ModuleList([
            ResidualUnit(int(nOutChannels), dilation=1), # in N out N
            ResidualUnit(int(nOutChannels), dilation=3), # in N out N
            ResidualUnit(int(nOutChannels), dilation=9) # in N out N
        ])
        self.conv = weight_norm(nn.ConvTranspose1d(nOutChannels*2, nOutChannels, kernel_size=kernel, stride=stride, padding=(kernel-stride) // 2))  # in N out 2N

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, LEAKY_RELU)
        
        for unit in self.res_units:
            x = unit(x)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, endChannels, base_width=32) -> None:
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(1, base_width, kernel_size=7, padding=get_padding(7)))
        self.ups = nn.ModuleList()
        upstrides = [8, 5, 3, 2,]

        multiplier = 1
        for i in range(len(upstrides)):
            multiplier *= 2
            self.ups.append(EncoderBlock(base_width * multiplier, stride=upstrides[i]))

        self.conv2 = weight_norm(nn.Conv1d(base_width*multiplier, endChannels, kernel_size=7, padding=get_padding(7)))
        nn.init.normal_(self.conv.weight, INIT_MEAN, INIT_STD)
        nn.init.normal_(self.conv2.weight, INIT_MEAN, INIT_STD)


    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, LEAKY_RELU)

        for unit in self.ups:
            x = unit(x)
        
        x = F.leaky_relu(x, LEAKY_RELU)
        x = self.conv2(x)
        x = F.tanh(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, endChannels, base_width=256) -> None:
        super().__init__()
        
        self.ups = nn.ModuleList()
        upstrides = [2, 3, 5, 8]
        upsample_kernel_sizes = [4, 7, 11, 16]
        
        self.conv = weight_norm(nn.Conv1d(endChannels, base_width, kernel_size=7, padding=get_padding(7)))

        for i in range(len(upstrides)):
            self.ups.append(DecoderBlock(base_width//(2**(i+1)), stride=upstrides[i], kernel = upsample_kernel_sizes[i]))

        self.conv2 = weight_norm(nn.Conv1d(base_width//(2**(len(upstrides))), 1, kernel_size=7, padding=get_padding(7)))
        nn.init.normal_(self.conv.weight, INIT_MEAN, INIT_STD)
        nn.init.normal_(self.conv2.weight, INIT_MEAN, INIT_STD)


    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, LEAKY_RELU)

        for unit in self.ups:
            x = unit(x)

        x = F.leaky_relu(x, LEAKY_RELU)
        x = self.conv2(x)
        x = F.tanh(x)
        return x
    