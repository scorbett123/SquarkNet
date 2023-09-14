from torch import nn
import torch

class MovingAverage(nn.Module):
    def __init__(self, context_len: int, intial_val=1) -> None:
        super().__init__()
        self.register_buffer("_window", torch.fill(torch.empty(context_len, dtype=torch.float32), intial_val))
        self.average = intial_val
        self._context_len = context_len

        weights = self.get_weights(context_len)
        self.register_buffer("_weights", weights)
        self._weight_sum = torch.sum(weights)
        self._length = 0

    @torch.no_grad()
    def update(self, new_value) -> torch.Tensor:
        self._length += 1
        self._window = torch.roll(self._window, 1)
        self._window[0] = new_value

        s = self._weight_sum if self._length >= self._context_len else torch.sum(self._weights[:self._length])
        result = torch.sum(self._window[:self._length] * self._weights[:self._length]) / s
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
    def __init__(self, *args, beta, **kwargs) -> None:  # the parameters here do a bit of trolling at points, when bugs look here
        self.beta = beta
        super().__init__(*args, **kwargs)

    def get_weights(self, context_len) -> torch.Tensor:
        weights = torch.linspace(0, context_len-1, context_len)
        weights =  self.beta ** weights
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