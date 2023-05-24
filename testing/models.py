from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
LEAKY_RELU = 0.2

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# paper https://arxiv.org/pdf/2009.02095.pdf
class ResidualUnit(nn.Module):
    """ Residual Unit, input and output are same dimension """
    def __init__(self, nChannels, dilation) -> None:
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(nChannels, nChannels, kernel_size=3, padding=get_padding(3, dilation), dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(nChannels, nChannels, kernel_size=1))

    def forward(self, x):
        xt = self.conv1(x)
        xt = F.leaky_relu(xt, LEAKY_RELU)
        xt = self.conv2(xt)
        xt = F.leaky_relu(xt, LEAKY_RELU)
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
        upstrides = [8, 3, 2,]

        multiplier = 1
        for i in range(len(upstrides)):
            multiplier *= 2
            self.ups.append(EncoderBlock(base_width * multiplier, stride=upstrides[i]))

        self.conv2 = weight_norm(nn.Conv1d(base_width*multiplier, endChannels, kernel_size=7, padding=get_padding(7)))


    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, LEAKY_RELU)

        for unit in self.ups:
            x = unit(x)
        
        x = self.conv2(x)
        x = F.leaky_relu(x, LEAKY_RELU)
        return x
    
class Decoder(nn.Module):
    def __init__(self, endChannels, base_width=256) -> None:
        super().__init__()
        
        self.ups = nn.ModuleList()
        upstrides = [2,3, 8]
        upsample_kernel_sizes = [4, 7, 16]
        
        self.conv = weight_norm(nn.Conv1d(endChannels, base_width, kernel_size=7, padding=get_padding(7)))

        size = base_width
        for i in range(len(upstrides)):
            self.ups.append(DecoderBlock(base_width//(2**(i+1)), stride=upstrides[i], kernel = upsample_kernel_sizes[i]))

        self.conv2 = weight_norm(nn.Conv1d(base_width//(2**(len(upstrides))), 1, kernel_size=7, padding=get_padding(7)))


    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, LEAKY_RELU)

        for unit in self.ups:
            x = unit(x)

        x = F.leaky_relu(x, LEAKY_RELU)
        x = self.conv2(x)
        x = F.tanh(x)
        return x