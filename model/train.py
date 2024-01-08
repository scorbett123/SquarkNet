import torch
import itertools
from models import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchaudio
import os
from loss import loss


class Trainer:
    def __init__(self, models: Models, train_loader: DataLoader, valid_loader: DataLoader, loss_gen: loss.LossGenerator, device="cpu", learning_rate=0.0002, betas=[0.5, 0.9], discrim_learning_rate=0.0002, gamma=0.99) -> None:
        self.models = models
        self._learning_rate = learning_rate
        self._discrim_learning_rate = discrim_learning_rate
        self._betas = betas
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._loss_gen = loss_gen
        self._device = device

        self._model_optimizer = torch.optim.Adam(itertools.chain(models.encoder.parameters(), models.decoder.parameters(), models.quantizer.parameters()), lr=learning_rate, betas=betas)
        self._discriminator_optimizer = torch.optim.Adam(models.discriminator.parameters(),  lr=discrim_learning_rate, betas=betas)

        self._scheduler_model = torch.optim.lr_scheduler.ExponentialLR(self._model_optimizer, gamma=gamma)
        self._scheduler_discrim = torch.optim.lr_scheduler.ExponentialLR(self._discriminator_optimizer, gamma=gamma)
        
        self.writer = SummaryWriter(log_dir="logs-t/")

        self.steps = 1 # start steps at 1 so that we don't run logging on first step

    def run_epoch(self):
        self.models.train()
        for x in tqdm(self._train_loader):
            self.models.train()  # can't be too careful
            x = x.to(self._device)
            y, q_loss = self.models(x)
            discrim_x, feature_x = self.models.discrim_forward(x)
            discrim_y, feature_y = self.models.discrim_forward(y)

            
            self._model_optimizer.zero_grad()
            loss = self._loss_gen.get_loss(x, y, discrim_y, feature_x, feature_y, q_loss)
            self.writer.add_scalar("test/main", loss.item(), self.steps)
            loss.backward(retain_graph=True)#(retain_graph=True)

            self._model_optimizer.step()
            
            self._discriminator_optimizer.zero_grad()
            discrim_x, _ = self.models.discrim_forward(x.detach())  # TODO figure out if there is a cleaner way to do this without requiring two runs through discrim
            discrim_y, _ = self.models.discrim_forward(y.detach())
            discrim_loss = self._loss_gen.get_discrim_loss(discrim_x, discrim_y)  # TODO only do this one in every n times
            
            self.writer.add_scalar("test/discrim", discrim_loss.item(), self.steps)
            discrim_loss.backward()
            if self.steps % 3 < 2:
                self._discriminator_optimizer.step()

            del x, discrim_x, discrim_y, discrim_loss, loss, feature_x, feature_y, y, q_loss  # we del them here so that when we deal with validation we don't care about them

            if self.steps % 100 == 0:
                with torch.no_grad():
                    self.models.quantizer.deal_with_dead()

            if self.steps % 10 == 0:
                self._loss_gen.plot(self.writer, self.steps)
            if self.steps % 2000 == 0:
                self.gen_samples(f"epoch{self.models.epochs}")
            if self.steps % 5000 == 0:
                self.gen_valid_losses()
            self.steps += 1

        self._scheduler_model.step()
        self._scheduler_discrim.step()


    @torch.no_grad()
    def gen_valid_losses(self):
        self.models.eval()
        total_loss = loss.ValidLoss(0,0,0,0,0,0)
        for x in self._valid_loader:
            x = x.to(self._device)
            y, quant_loss = self.models(x)
            discrim_x, feat_x = self.models.discrim_forward(x.detach())  # TODO figure out if there is a cleaner way to do this without requiring two runs through discrim
            discrim_y, feat_y = self.models.discrim_forward(y.detach())
            
            total_loss += self._loss_gen.get_valid_loss(x, y, discrim_x, discrim_y, feat_x, feat_y, quant_loss)
        
        total_loss.write(self.writer, self.steps)
        self.models.train()


    def save_model(self, folder_name):
        self.models.save(folder_name)


    def gen_samples(self, folder_name):
        self.models.eval()
        os.makedirs(f"samples/{folder_name}", exist_ok = True) 
        for i, case in enumerate(self._valid_loader):
            if i > 5:
                break
            output, _ = self.models(case.to("cuda"))
            torchaudio.save(f"samples/{folder_name}/{i}-clean.wav", case[0], sample_rate=16000)
            torchaudio.save(f"samples/{folder_name}/{i}-encoded.wav", output.cpu()[0], sample_rate=16000)
        self.models.train()