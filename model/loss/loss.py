from models import Models
from loss_functions import *

class LossGenerator():  # TODO more params such as weights etc.
    def __init__(self) -> None:
        self.discrim_ad_loss = DiscriminatorAdversairialLoss(1)

        self.discrim_loss = DiscriminatorLoss(1)
        self.recon_loss = ReconstructionLoss(1)
        self.quantization_loss = SetLoss("quantization_loss", 1)
        self.whisper_loss = WhisperLoss()
        # TODO feature loss

    def get_loss(self, x, y, discrim_x, discrim_y, quantization_loss):
        q = self.quantization_loss.get_value(quantization_loss)
        r = self.recon_loss.get_value(x, y)
        w = self.recon_loss.get_value(x, y)
        d = self.discrim_loss.get_value(discrim_x, discrim_y)
        return q + r + w + d

    def get_discrim_loss(self, discrim_x, discrim_y):
        return self.discrim_ad_loss.get_value(discrim_x, discrim_y)