from models import Models
from model.loss.loss_functions import *
from dataclasses import dataclass, asdict

class LossGenerator():  # TODO more params such as weights etc.
    def __init__(self, context_length, batch_size, device="cpu") -> None:
        self.discrim_ad_loss = DiscriminatorAdversairialLoss(1).to(device)

        self.discrim_loss = DiscriminatorLoss(2).to(device)
        self.feature_loss = FeatureLoss(3).to(device)
        self.recon_loss = ReconstructionLoss(0.005, 2).to(device)  # first value should be as low as needed for silence to be silent.
        self.quantization_loss = SetLoss("quantization_loss", 2).to(device)
        self.whisper_loss = WhisperLoss(context_length, batch_size,3).to(device)

        self.losses = [self.discrim_ad_loss, self.discrim_loss, self.feature_loss, self.recon_loss, self.quantization_loss, self.whisper_loss]
        # TODO feature loss

    def get_loss(self, x, y, discrim_y, feat_x, feat_y, quantization_loss):
        q = self.quantization_loss.get_value(quantization_loss)
        r = self.recon_loss.get_value(x, y)
        w = self.whisper_loss.get_value(x, y)
        d = self.discrim_loss.get_value(discrim_y)
        f = self.feature_loss.get_value(feat_x, feat_y)
        return q + r + w + d + f

    def get_discrim_loss(self, discrim_x, discrim_y):
        return self.discrim_ad_loss.get_value(discrim_x, discrim_y)
    
    def plot(self, writer, steps):
        for loss in self.losses:
            loss.plot(writer, steps)

    def get_valid_loss(self, x, y, discrim_x, discrim_y, feat_x, feat_y, quantization_loss):
        q = self.quantization_loss.get_raw_value(quantization_loss)
        r = self.recon_loss.get_raw_value(x, y)
        w = self.whisper_loss.get_raw_value(x, y)
        d = self.discrim_loss.get_raw_value(discrim_y)
        f = self.feature_loss.get_raw_value(feat_x, feat_y)
        da = self.discrim_ad_loss.get_raw_value(discrim_x, discrim_y)
        return ValidLoss(q.item(), r.item(), w.item(), d.item(), f.item(), da.item())
    
@dataclass
class ValidLoss:
    quantization_loss: float
    recon_loss: float
    whisper_loss: float
    discrim_loss: float
    feature_loss: float
    discrim_adversairial_loss: float

    def __add__(self, other):
        return ValidLoss(self.quantization_loss + other.quantization_loss,
                    self.recon_loss + other.recon_loss,
                    self.whisper_loss + other.whisper_loss,
                    self.discrim_loss + other.discrim_loss,
                    self.feature_loss + other.feature_loss,
                    self.discrim_adversairial_loss + other.discrim_adversairial_loss)
    
    def write(self, writer: SummaryWriter, steps: int):
        ad = asdict(self)
        for key in ad:
            writer.add_scalar(f"valid/{key}", ad[key], steps)

