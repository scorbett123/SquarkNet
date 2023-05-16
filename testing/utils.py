import torch
import matplotlib.pyplot as plt

def plot_waveform(waveform, file="matplotlib.png"):
    waveform = waveform.numpy()[0]
    print(waveform.shape)
    ax = plt.subplot()
    ax.set_ylim(bottom=-1, top=1)
    ax.plot(torch.arange(0, waveform.shape[0]), waveform)
    plt.savefig(file)  # developing using wsl, but want to work on normal linux, so not messing with backends to make show() work