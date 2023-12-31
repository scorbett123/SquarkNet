from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from model.loss.discriminator_model import MultiScaleSTFTDiscriminator
from model import vq
import math
import os
import torch
import gzip
import hashlib
import json
import itertools, functools
import io
LEAKY_RELU = 0.2
INIT_MEAN = 0.0
INIT_STD = 0.01


class Models(nn.Module):
    def __init__(self, n_channels, nbooks, ncodes, epochs=0, sftf_scales=[1024, 512, 256], upstrides=[2,4,6,8], device="cpu", discrim=True) -> None:  # improve type hints here, probably make a separate quantizer class
        super().__init__()
        self.n_channels = n_channels
        self.nbooks = nbooks
        self.ncodes = ncodes
        self.stft_scales = sftf_scales
        self.epochs = epochs
        self.upstrides = upstrides

        self.encoder = Encoder(n_channels, upstrides=upstrides).to(device)
        self.quantizer = vq.RVQ(nbooks, ncodes, n_channels).to(device)
        self.decoder = Decoder(n_channels, upstrides=upstrides).to(device)
        if discrim:
            self.discriminator = MultiScaleSTFTDiscriminator(scales=sftf_scales).to(device)
        else:
            self.discriminator = None

    def forward(self, x):
        """ return in the form y, q_loss """
        encoded  = self.encoder(x)

        encoded = torch.transpose(encoded, -1, -2)
        after_q, _, q_loss  = self.quantizer(encoded)
        after_q = torch.transpose(after_q, -1, -2)
        y = self.decoder(after_q)
        return y, q_loss
    
    def encode(self, x):
        encoded  = self.encoder(x)

        encoded = torch.transpose(encoded, -1, -2)
        output = self.quantizer.encode(encoded)
        return output

    def decode(self, x):
        after_q = self.quantizer.decode(x)
        after_q = torch.transpose(after_q, -1, -2)
        y = self.decoder(after_q)
        return y
    
    def discrim_forward(self, x):
        return self.discriminator(x)
    
    def save(self, file_name):
        os.makedirs(f"{''.join(file_name.split('/')[:-1])}", exist_ok = True) 
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        buffer.seek(0)
        model_files = buffer.read()
        model_hash = hashlib.md5(model_files).digest()

        params = {
            "n_channels": self.n_channels,
            "ncodes": self.ncodes,
            "nbooks": self.nbooks,
            "upstrides": self.upstrides,
            "epochs": self.epochs
        }

        result = {
            "model_hash": model_hash,
            "params": params,
            "models": model_files
        }
        
        torch.save(result, f"{file_name}")

    
    @property
    def ctx_len(self):
        return functools.reduce(lambda x, y : x*y, self.upstrides)
    
    @property
    def hash(self):
        """ Really slow, shouldn't be used frequently """
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        buffer.seek(0)
        model_files = buffer.read()
        model_hash = hashlib.md5(model_files).digest()
        hash_as_int = int.from_bytes(model_hash, byteorder="big",signed=False)
        return hash_as_int

    
    def load(folder_name, device="cpu"):
        try:
            m = torch.load(folder_name)
        except IOError:
            raise Exception("File doesn't exist")

        if hashlib.md5(m["models"]).digest() != m["model_hash"]:
            raise Exception("Invalid hash")
        
        model_statedict = torch.load(io.BytesIO(m["models"]))
        params = m["params"]
        models = Models(params["n_channels"], params["nbooks"], params["ncodes"], epochs=params["epochs"], device=device)
        models.load_state_dict(model_statedict)
        return models


def get_padding(kernel_size, dilation=1):
    return ((kernel_size-1) * dilation) // 2

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
    """ Decoder block, requires input size to be nOutChannels / 2 """
    def __init__(self, nOutChannels, stride) -> None:
        super().__init__()
        self.res_units = nn.ModuleList([
            ResidualUnit(int(nOutChannels), dilation=1), # in N out N
            ResidualUnit(int(nOutChannels), dilation=3), # in N out N
            ResidualUnit(int(nOutChannels), dilation=9) # in N out N
        ])
        self.conv = weight_norm(nn.ConvTranspose1d(nOutChannels*2, nOutChannels, kernel_size=2*stride, stride=stride, padding=math.ceil((stride) / 2)))  # in 2N out N

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, LEAKY_RELU)
        
        for unit in self.res_units:
            x = unit(x)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, endChannels, upstrides=[2,4,6,8], base_width=32) -> None:
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(1, base_width, kernel_size=7, padding=get_padding(7)))
        self.ups = nn.ModuleList()

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
        y = F.tanh(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, endChannels, upstrides=[2,4,6,8], base_width=256) -> None:
        super().__init__()
        
        self.ups = nn.ModuleList()
        dowwnstrides = upstrides[::-1]
        self.conv = weight_norm(nn.Conv1d(endChannels, base_width, kernel_size=7, padding=get_padding(7)))

        for i in range(len(dowwnstrides)):
            self.ups.append(DecoderBlock(base_width//(2**(i+1)), stride=dowwnstrides[i]))

        self.conv2 = weight_norm(nn.Conv1d(base_width//(2**(len(dowwnstrides))), 1, kernel_size=7, padding=get_padding(7)))
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
    