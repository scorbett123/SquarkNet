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
    axarr[0].imshow(spectrogram1.log2()[0,:,:].to("cpu").numpy())
    axarr[1].imshow(spectrogram2.log2()[0,:,:].to("cpu").numpy())
    f.savefig(file)  # developing using wsl, but want to work on normal linux, so not messing with backends to make show() work
    plt.close()