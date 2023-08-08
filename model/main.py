import torch
import datasets
import models
from torch.utils.data import DataLoader
import vq
import train_new
from model.loss.loss import LossGenerator

def main():
    context_length = 384*32
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = datasets.TrainSpeechDataset(context_length)
    valid_loader = datasets.ValidateSpeechDataset(48)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_loader, batch_size=1)  # TODO increase batch size here, just 1 for testing

    loss_gen = LossGenerator(context_length, batch_size, device=device)

    # encoder.load_state_dict(torch.load("logs/encoder.state"))
    # decoder.load_state_dict(torch.load("logs/decoder.state"))

    # intermediate channels, nbooks, ncodes, discrim steps
    m = models.Models(192, 4, 1024, device=device)
    trainer = train_new.Trainer(m, train_dataloader, valid_loader, loss_gen, device=device)
    i = 0
    while True:
        trainer.run_epoch(i)
        i += 1
        #trainer.save_model(f"epoch{i}")
        print("Epoch {i} done")

if __name__ == "__main__":
    main()