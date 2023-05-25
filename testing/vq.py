import torch.nn
import torch.nn.functional as F

class VQ(torch.nn.Module):

    def __init__(self, codebook_size, codeword_size) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codeword_size = codebook_size

        self.embedding = torch.nn.Parameter(torch.zeros((codebook_size, codeword_size)))
        torch.nn.init.normal_(self.embedding)  # TODO Use k-means on first batch to attempt to give better initialization, also add dead 


    def forward(self, x):
        print(x)
        x = x.transpose(1, 2)  # B C T -> B T C
        dist = torch.sum(x**2, dim=2, keepdim=True) + torch.sum(self.embedding**2, dim=1) - 2.0 * torch.matmul(x, self.embedding.t())
        print(dist)
        assert 1== 0
        vals, indexes = torch.min(dist, dim=2) # this isn't differentiable, so need to add in loss later (passthrough loss)

        #  Two lines below are just normal style embedding
        one_hot = F.one_hot(indexes, num_classes=self.codebook_size).float()
        values = torch.matmul(one_hot, self.embedding)

        loss1 = F.mse_loss(x.detach(), values)  # only train the embedding
        loss2 = F.mse_loss(x, values.detach())  # only train the encoder
        values = values.transpose(1, 2)  # B T C to B C T
        print(indexes)
        return values, indexes, loss1 + loss2

    
    def initialise(self, x): # for now this code doesn't need to be efficient as only run once. My own take on k_means
        vectors = x.flatten(end_dim=1)
        unique_vectors = vectors.unique(dim=0)

        random_idx = torch.randperm(vectors.size(dim=0))[:self.codebook_size]
        if random_idx.shape[0] != self.codebook_size:
            print("possible error with not enough unique vectors")

        kmeans_centroids = vectors[random_idx] # initialise the centroids for kmeans on randomly selected points from data

        for i in range(len(x)): # kmeans iter
            # each vector finds its closest centroid
            distances = torch.sum(vectors**2, dim=1, keepdim=True) + torch.sum(kmeans_centroids**2, dim=1) - 2.0 * torch.matmul(vectors, kmeans_centroids)
            dists, indexes = torch.min(distances, dim=1)

            # for each centroid find the average position
            for i in range(self.codebook_size):
                cluster_vals = torch.nonzero(indexes == i)



class RVQ(torch.nn.Module):
    """ Simple residual vector quantizer, Algo. 1 in https://arxiv.org/pdf/2107.03312.pdf"""
    def __init__(self, n_residuals, codebook_size, codeword_size) -> None:
        super().__init__()
        self.n_residals = n_residuals
        self.codebook_size = codebook_size
        self.codeword_size = codeword_size
        self.quantizers = torch.nn.ModuleList([VQ(self.codebook_size, self.codeword_size) for i in range(n_residuals)])

    def forward(self, x):
        y_hat = torch.zeros_like(x)
        residual =  x
        loss = 0
        for q in self.quantizers:
            q_values, _, q_loss = q(residual)  # in this we don't actually care about the indices
            residual = residual - q_values  # inplace operators mess with pytorch, so we can't use them
            y_hat = y_hat + q_values  # see above
            loss = loss+q_loss  # see above

        return y_hat, loss

    


# x = VQ(5, 5)
# print(x.parameters())
# optimizer = torch.optim.SGD(x.parameters(), lr=0.01, momentum=0.9)

# for i in range(1000):
#     values, indexes, loss = x(torch.tensor([[[1, 0, 1, 1, 0], [0, 1, 2, 3, 1]]], dtype=torch.float32))
#     print(loss)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()