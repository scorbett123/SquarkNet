from torch import nn
import torch

class MovingAverage(nn.Module):
    def __init__(self, context_len: int) -> None:
        super().__init__()
        self.register_buffer("window", torch.ones((context_len)))
        self.average = 1

        weights = self.get_weights()
        self.register_buffer("weights", weights)
        self.weight_sum = torch.sum(weights)

    @torch.no_grad()
    def update(self, new_value):
        self.window = torch.roll(self.window, 1)
        self.window[0] = new_value
        result = torch.sum(self.window * self.weights) / self.weight_sum
        self.average = result
        return result
    
    def get_weights(self, context_len) -> torch.Tensor:
        raise NotImplemented


class EMA(MovingAverage):
    """
    Exponential Moving Average

    The weight of each element decreases with the form 1/n. IDK if this is actually exponential moving average, however it was the first thing
    I found and I'm about to run out of mobile data.

    """

    def get_weights(self, context_len) -> torch.Tensor:
        weights = torch.linspace(1, context_len, context_len)
        weights =  1/weights
        return weights
   
    
class SMA(MovingAverage):
    """
    Simple moving average

    The weight of all elements in the context length is equal

    This could probably be implemented more efficiently, however this is pretty clean. 
    Only thing is a useless elementwise multiply (we don't care about slow initialisation), but that is at training only.
    """

    def get_weights(self, context_len) -> torch.Tensor:
        weights = torch.ones(context_len)
        return weights