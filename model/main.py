import torch
from model import datasets
import models
from torch.utils.data import DataLoader
import vq
import model.train as train
from model.loss.loss import LossGenerator

def main():
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    m = models.Models(192, 4, 1024, upstrides=[2,4,6,8], device=device)
    # m = models.Models.load("logs-t/epoch12/models.saved", device=device)
    context_length = m.ctx_len*32

    train_data = datasets.CommonVoice(context_length)
    valid_loader = datasets.CommonVoice(context_length, "test", length = 100)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, num_workers=8)

    loss_gen = LossGenerator(context_length, batch_size, device=device)

    #m = models.Models.load("logs-t/epoch23").to(device)
    trainer = train.Trainer(m, train_dataloader, valid_loader, loss_gen, device=device, learning_rate=0.00005)
    while True:
        trainer.run_epoch()
        trainer.save_model(f"epoch{m.epochs}")
        m.epochs += 1
        print(f"Epoch {m.epochs} starting")

if __name__ == "__main__":
    main()