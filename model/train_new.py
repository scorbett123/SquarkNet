import torch
import itertools
from models import *
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, models: Models, train_loader, valid_loader, loss_gen, learning_rate=0.0002, betas=[0.5, 0.9], discrim_learning_rate=0.002, gamma=0.98) -> None:
        self.models = models
        self.learning_rate = learning_rate
        self.discrim_learning_rate = discrim_learning_rate
        self.betas = betas
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_gen = loss_gen

        self.model_optimizer = torch.optim.Adam(itertools.chain(models.encoder.parameters(), models.decoder.parameters(), models.quantizer.parameters()), lr=learning_rate, betas=betas)
        self.discriminator_optimizer = torch.optim.Adam(models.discriminator.parameters(),  lr=discrim_learning_rate, betas=betas)

        self.scheduler_model = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer, gamma=gamma)
        self.scheduler_discrim = torch.optim.lr_scheduler.ExponentialLR(self.discriminator_optimizer, gamma=gamma)
        
        self.epoch_num = 0

    def run_epoch(self):
        self.models.train()
        for x in self.train_loader:
            y, q_loss = self.models(x)
            discrim_x = self.models.discrim_forward(x)
            discrim_y = self.models.discrim_forward(y)

            self.discriminator_optimizer.zero_grad()
            discrim_loss = self.loss_gen.get_discrim_loss(discrim_x, discrim_y)  # TODO only do this one in every n times
            discrim_loss.backward()  # TODO see if there is a way to make this stop backpropogating after discrims, and not through the whole model
            self.discriminator_optimizer.step()
            
            self.model_optimizer.zero_grad()
            loss = self.loss_gen.get_loss(x, y, discrim_x, discrim_y, q_loss)
            loss.backward()
            self.model_optimizer.step()


        self.epoch_num += 1
        self.scheduler_model.step()
        self.scheduler_discrim.step()