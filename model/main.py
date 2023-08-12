import torch
from model import datasets
import models
from torch.utils.data import DataLoader
import vq
import train_new
from model.loss.loss import LossGenerator

def main():
    context_length = 384*32
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = datasets.TrainSpeechDataset(context_length, length=100)
    valid_loader = datasets.ValidateSpeechDataset(48)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_loader, batch_size=1)  # TODO increase batch size here, just 1 for testing

    loss_gen = LossGenerator(context_length, batch_size, device=device)

    # m = models.Models(192, 4, 1024, device=device)
    m = models.Models.load("logs-t/epoch6").to(device)
    trainer = train_new.Trainer(m, train_dataloader, valid_loader, loss_gen, device=device)
    while True:
        trainer.run_epoch()
        m.epochs += 1
        trainer.save_model(f"epoch{m.epochs}")
        print(f"Epoch {m.epochs} done")

if __name__ == "__main__":
    main()