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
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = models.Encoder(256).to(device)
quantizer = vq.RVQ(5, 1024, 256).to(device)
decoder = models.Decoder(256).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compresser")
    parser.add_argument("filename")

    if parser.filename.endswith("sc"):

    else:
        
