import torch
import itertools
from models import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchaudio


class Trainer:
    def __init__(self, models: Models, train_loader, valid_loader, loss_gen, device="cpu", learning_rate=0.0002, betas=[0.5, 0.9], discrim_learning_rate=0.0002, gamma=0.98) -> None:
        self.models = models
        self.learning_rate = learning_rate
        self.discrim_learning_rate = discrim_learning_rate
        self.betas = betas
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_gen = loss_gen
        self.device = device

        self.model_optimizer = torch.optim.Adam(itertools.chain(models.encoder.parameters(), models.decoder.parameters(), models.quantizer.parameters()), lr=learning_rate, betas=betas)
        self.discriminator_optimizer = torch.optim.Adam(models.discriminator.parameters(),  lr=discrim_learning_rate, betas=betas)

        self.scheduler_model = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer, gamma=gamma)
        self.scheduler_discrim = torch.optim.lr_scheduler.ExponentialLR(self.discriminator_optimizer, gamma=gamma)
        
        self.epoch_num = 0
        self.writer = SummaryWriter(log_dir="logs-t/")

        torch.autograd.set_detect_anomaly(True)
        self.steps = 1 # start steps at 1 so that we don't run logging on first step

    def run_epoch(self):
        self.models.train()
        for x in tqdm(self.train_loader):
            self.models.train()  # can't be too careful
            x = x.to(self.device)
            y, q_loss = self.models(x)
            discrim_x, feature_x = self.models.discrim_forward(x)
            discrim_y, feature_y = self.models.discrim_forward(y)

            
            self.model_optimizer.zero_grad()
            loss = self.loss_gen.get_loss(x, y, discrim_y, feature_x, feature_y, q_loss)
            self.writer.add_scalar("test/main", loss.item(), self.steps)
            loss.backward(retain_graph=True)#(retain_graph=True)

            self.model_optimizer.step()
            
            self.discriminator_optimizer.zero_grad()
            discrim_x, _ = self.models.discrim_forward(x.detach())  # TODO figure out if there is a cleaner way to do this without requiring two runs through discrim
            discrim_y, _ = self.models.discrim_forward(y.detach())
            discrim_loss = self.loss_gen.get_discrim_loss(discrim_x, discrim_y)  # TODO only do this one in every n times
            
            self.writer.add_scalar("test/discrim", discrim_loss.item(), self.steps)
            discrim_loss.backward()
            if self.steps %3 < 2:
                self.discriminator_optimizer.step()

            if self.steps % 120 == 0:
                with torch.no_grad():
                    self.models.quantizer.deal_with_dead()

            if self.steps % 25 == 0:
                self.loss_gen.plot(self.writer, self.steps)
            if self.steps % 250 == 0:
                self.gen_samples()
            self.steps += 1



        self.epoch_num += 1
        self.scheduler_model.step()
        # self.scheduler_discrim.step()

    
    def save_model(self):
        torch.save(self.models.encoder.state_dict(), "logs-t/encoder.state")
        torch.save(self.models.decoder.state_dict(), "logs-t/decoder.state")
        torch.save(self.models.quantizer.state_dict(), "logs-t/quantizer.state")


    def gen_samples(self):
        self.models.eval()
        for i, case in enumerate(self.valid_loader):
            if i > 5:
                break
            output, _ = self.models(case.to("cuda"))
            torchaudio.save(f"samples/{i}-clean.wav", case[0], sample_rate=16000)
            torchaudio.save(f"samples/{i}-encoded.wav", output.cpu()[0], sample_rate=16000)
        self.models.train()