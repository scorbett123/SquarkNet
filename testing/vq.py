from torch import nn
import torch.nn.functional as F
import random
import torch

class VQ(nn.Module):

    def __init__(self, codebook_size, codeword_size, n=1, beta=0.1) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codeword_size = codebook_size
        self.encoder_fit_vq_factor = beta

        self.embedding = nn.Parameter(torch.zeros((codebook_size, codeword_size)))
        self.embedding.data.uniform_(-1.0 / n, 1.0 / n) # TODO Use k-means on first batch to attempt to give better initialization, also add dead 


    def forward(self, x):
        dist = torch.sum(x**2, dim=2, keepdim=True) + torch.sum(self.embedding**2, dim=1) - 2.0 * torch.matmul(x, self.embedding.t())
        vals, indexes = torch.min(dist, dim=2) # this isn't differentiable, so need to add in loss later (passthrough loss)

        #  Two lines below are just normal style embedding
        one_hot = F.one_hot(indexes, num_classes=self.codebook_size).float()
        values = torch.matmul(one_hot, self.embedding)

        loss1 = F.mse_loss(x.detach(), values)  # only train the embedding
        loss2 = F.mse_loss(x, values.detach())  # only train the encoder

        values = x + (values -x).detach()
        
        return values, indexes, loss1 + loss2 * self.encoder_fit_vq_factor

    
    def initialise(self, x): # for now this code doesn't need to be efficient as only run once. My own take on k_means
        return # figure out the issues in this
        x = x.transpose(1, 2)  # B C T -> B T C
        vectors = x.flatten(end_dim=1)
        unique_vectors = vectors.unique(dim=0)
        print(unique_vectors.shape, vectors.shape)

        random_idx = torch.randperm(unique_vectors.size(dim=0))[:self.codebook_size]
        if random_idx.shape[0] != self.codebook_size:
            print("possible error with not enough unique vectors")
            raise Exception("Not enough vectors")

        kmeans_centroids = unique_vectors[random_idx] # initialise the centroids for kmeans on randomly selected points from data

        for i in range(2): # kmeans iter
            # each vector finds its closest centroid
            distances = torch.sum(vectors**2, dim=1, keepdim=True) + torch.sum(kmeans_centroids**2, dim=1) - 2.0 * torch.matmul(vectors, kmeans_centroids.t())
            dists, indexes = torch.min(distances, dim=1)

            # for each centroid find the average position
            for i in range(self.codebook_size):
                cluster_indices = torch.nonzero(indexes == i).squeeze(1)
                if cluster_indices.shape[0] > 0:
                    one_hot_eq = F.one_hot(cluster_indices, num_classes=vectors.shape[0]).float()
                    cluster_vals = torch.matmul(one_hot_eq, vectors)
                    average = cluster_vals.sum(dim=0) / cluster_indices.shape[0]
                    
                    kmeans_centroids[i] = average
                else:
                    print(dists.shape)
                    random_vector, _ = torch.max(dists, dim=0) # instead of random initialization as in the paper, initalize to whereever is worst represented
                    dists = dists[dists[:] == random_vector]
                    kmeans_centroids[i] = random_vector
        with torch.no_grad():
            self.embedding[:] = kmeans_centroids[:]
    



class RVQ(torch.nn.Module):
    """ Simple residual vector quantizer, Algo. 1 in https://arxiv.org/pdf/2107.03312.pdf"""
    def __init__(self, n_residuals, codebook_size, codeword_size) -> None:
        super().__init__()
        self.n_residals = n_residuals
        self.codebook_size = codebook_size
        self.codeword_size = codeword_size
        self.quantizers = nn.ModuleList([VQ(self.codebook_size, self.codeword_size) for i in range(n_residuals)])

    def forward(self, x):

        y_hat = torch.zeros_like(x)
        residual =  x
        loss = 0
        for i, q in enumerate(self.quantizers):
            q_values, indices, l = q(residual)  # in this we don't actually care about the indices
            residual = residual - q_values  # inplace operators mess with pytorch, so we can't use them
            y_hat = y_hat + q_values  # see above
            loss = loss+l  # see above

        loss = loss / len(self.quantizers)
        
        y_hat = x + (y_hat - x).detach() # maintains gradients in x, but removes the ones we don't want in y_hat
        return y_hat, loss
    
    def initialise(self, x):
        residual = x
        for q in self.quantizers:
            q.initialise(residual)
            q_values, _, _ = q(residual)
            residual = residual - q_values
