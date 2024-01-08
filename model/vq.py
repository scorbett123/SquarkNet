from torch import nn
import torch.nn.functional as F
import random
import torch


class VectorCache():
    def __init__(self, length) -> None:
        self._length = length
        self._vectors = []  # if I need to I could implement a proper circular queue here

    def add_vector(self, new_vector: torch.Tensor):
        self._vectors.append(new_vector)
        if len(self._vectors) > self._length:
            self._vectors.pop(0)

    def concat(self) -> torch.Tensor:
        return torch.cat(self._vectors, dim=0)

class VQ(nn.Module):
    def __init__(self, codebook_size, codeword_size, n=1, beta=0.2) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codeword_size = codeword_size
        self.encoder_fit_vq_factor = beta

        self.embedding = nn.Parameter(torch.zeros((codebook_size, codeword_size)))
        self.embedding.data.uniform_(-1.0 / (n**2), 1.0 / (n**2))

        usages = torch.zeros((codebook_size), requires_grad=False)
        self.register_buffer('usages', usages)
        self.cache = VectorCache(20)


    def apply_dead(self):
        indices_of_dead = self.usages == 0 # WARNING this doesn't actually give a list of indices, but a list of true / false values
        indices_of_alive = self.usages > 0

        self.frozen_kmeans(self.cache.concat(), indices_of_alive)
        
        self.usages[:] = torch.zeros((self.codebook_size), requires_grad=False)
    
    def decode(self, indexes):
        one_hot = F.one_hot(indexes, num_classes=self.codebook_size).float()
        values = torch.matmul(one_hot, self.embedding)
        return values

    @torch.no_grad()
    def frozen_kmeans(self, cached, fronzen_state, kmeans_iters=5):
        frozen_length = torch.sum(fronzen_state).to(torch.int32)  # shorthand for count boolean
        unfrozen_length = fronzen_state.shape[0] - frozen_length
        unique_vecs = cached.unique(dim=0)

        random_idx = torch.randperm(unique_vecs.size(dim=0))[:unfrozen_length]
        self.embedding[torch.logical_not(fronzen_state)] = cached[random_idx]

        ## Initialisaation complete move on to algorithm

        for _ in range(kmeans_iters):
            distances = torch.sum(cached**2, dim=1, keepdim=True) + torch.sum(self.embedding, dim=1) - 2.0 * torch.matmul(cached, self.embedding.t())  # calculate the distances from each centroid to the value
            dists, indexes = torch.min(distances, dim=1)

            counts = torch.bincount(indexes, minlength=fronzen_state.shape[0])  # count the number in each bin
            counts[fronzen_state] = -1  # make sure to mark as frozen

            ### First deal with those with no nearby points ###
            # These have to be dealt with separately as otherwise would get NaN
            #  get the indices of the points furthest away from their centroid
            indices = torch.sort(dists).indices[:torch.sum(counts==0)]  # torch.sum(Tensor[bool]) is short hand for count number of trues
            self.embedding[counts == 0] = cached[indices]

            ### Then deal with the means ###
            oh = F.one_hot(indexes, num_classes=fronzen_state.shape[0]).to(torch.float32).t() 

            amount_of_each = torch.sum(oh, dim=1)
            total_of_each = torch.matmul(oh, cached)
            result_means = total_of_each / torch.unsqueeze(amount_of_each, 1)
            # be careful with result means, where counts = 0 result means > 0

            self.embedding[counts > 0] = result_means[counts > 0]
    
    
    def decode(self, indexes):
        one_hot = F.one_hot(indexes, num_classes=self.codebook_size).float()
        values = torch.matmul(one_hot, self.embedding)
        return values

    def forward(self, x):
        self.cache.add_vector(x.reshape(-1, self.codeword_size).detach())
        dist = torch.sum(x**2, dim=-1, keepdim=True) + torch.sum(self.embedding**2, dim=1) - 2.0 * torch.matmul(x, self.embedding.t()) # x^2 + y^2 - 2xy
        vals, indexes = torch.min(dist, dim=-1) # this isn't differentiable, so need to add in loss later (passthrough loss)
        
        #  Two lines below are just normal style embedding
        one_hot = F.one_hot(indexes, num_classes=self.codebook_size).float()
        values = torch.matmul(one_hot, self.embedding)

        loss1 = F.mse_loss(x.detach(), values)  # only train the embedding
        loss2 = F.mse_loss(x, values.detach())  # only train the encoder

        values = x + (values - x).detach()

        if self.training:
            self.usages = self.usages + torch.sum(one_hot.reshape(-1, self.codebook_size), dim=-2)
        
        return values, indexes, loss1 + loss2 * self.encoder_fit_vq_factor
    


class RVQ(torch.nn.Module):
    """ Simple residual vector quantizer, Algo. 1 in https://arxiv.org/pdf/2107.03312.pdf"""
    def __init__(self, n_residuals, codebook_size, codeword_size, bypass_factor=0.5) -> None:
        super().__init__()
        self.n_residals = n_residuals
        self.codebook_size = codebook_size
        self.codeword_size = codeword_size
        self.quantizers = nn.ModuleList([VQ(self.codebook_size, self.codeword_size) for i in range(n_residuals)])

        self.bypass_factor = bypass_factor

    def forward(self, x):
        y_hat = torch.zeros_like(x)
        residual =  x
        loss = 0
        indices = None
        for i, q in enumerate(self.quantizers):
            q_values, q_i, l = q(residual)  # in this we don't actually care about the indices
            residual = residual - q_values  # inplace operators mess with pytorch, so we can't use them
            y_hat = y_hat + q_values  # see above
            loss = loss+l  # see above
            q_i = q_i.unsqueeze(-1)
            indices = q_i if indices == None else torch.cat((indices, q_i), dim=-1)

        loss = loss / len(self.quantizers)
        
        y_hat = x + (y_hat - x).detach() # maintains gradients in x, but removes the ones we don't want in y_hat
        
        return y_hat, indices, loss
    
    def encode(self, data):
        y_hat, indices, loss = self.forward(data)
        return indices

    def decode(self, indices):
        result = 0.0
        for i, q in enumerate(self.quantizers):
            result += q.decode(indices[..., i])
        return result

    def deal_with_dead(self):
        for q in self.quantizers:
            q.apply_dead()