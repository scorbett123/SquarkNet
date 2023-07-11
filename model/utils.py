import torch
import matplotlib.pyplot as plt

def plot_waveform(waveform, file="matplotlib.png"):
    waveform = waveform.to("cpu").numpy()[0]
    print(waveform.shape)
    ax = plt.subplot()
    ax.set_ylim(bottom=-1, top=1)
    ax.plot(torch.arange(0, waveform.shape[0]), waveform)
    plt.savefig(file)  # developing using wsl, but want to work on normal linux, so not messing with backends to make show() work

def plot_spectrograms(spectrogram1, spectrogram2, file="matplotlib.png"):
    plt.figure()

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(spectrogram1.log2()[0,:,:500].to("cpu").numpy(), aspect='auto')
    axarr[1].imshow(spectrogram2.log2()[0,:,:500].to("cpu").numpy(), aspect='auto')
    f.savefig(file)  # developing using wsl, but want to work on normal linux, so not messing with backends to make show() work
    plt.close()

def norm(tensor):  # TODO check this actually does what I think it does
    l, _ = torch.max(torch.abs(tensor), dim=-1)
    return (tensor / l) * 0.80