from models import Models
from model.loss.loss_functions import *

class LossGenerator():  # TODO more params such as weights etc.
    def __init__(self, context_length, batch_size, device="cpu") -> None:
        self.discrim_ad_loss = DiscriminatorAdversairialLoss(1).to(device)

        self.discrim_loss = DiscriminatorLoss(3).to(device)
        self.feature_loss = FeatureLoss(3).to(device)
        self.recon_loss = ReconstructionLoss(0.01, 1).to(device)
        self.quantization_loss = SetLoss("quantization_loss", 1).to(device)
        self.whisper_loss = WhisperLoss(context_length, batch_size,2).to(device)

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