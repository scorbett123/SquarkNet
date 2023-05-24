import torch.nn
import torch.nn.functional as F

class VQ(torch.nn.Module):

    def __init__(self, codebook_size, codeword_size) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codeword_size = codebook_size

        self.embedding = torch.nn.Parameter(torch.randn((codebook_size, codeword_size)))
        print(self.embedding)


    def forward(self, x):
        #x = x.transpose(1, 2)  # B C T -> B T C
        print()
        dist = torch.sum(x**2, dim=2, keepdim=True) + torch.sum(self.embedding**2, dim=1) - 2.0 * torch.matmul(x, self.embedding.t())
        vals, indexes = torch.min(dist, dim=2) # this isn't differentiable, so need to add in loss later (passthrough loss)

        #  Two lines below are just normal style embedding
        one_hot = F.one_hot(indexes, num_classes=self.codebook_size).float()
        values = torch.matmul(one_hot, self.embedding)

        loss1 = F.mse_loss(x.detach(), values)  # only train the embedding
        loss2 = F.mse_loss(x, values.detach())  # only train the encoder
        return values, indexes, loss1 + loss2

class RVQ(torch.nn.Module):

    def __init__(self, n_residuals, codebook_size, codeword_size) -> None:
        super().__init__()
        self.n_residals = n_residuals
        self.codebook_size = codebook_size
        self.codeword_size = codeword_size

    def forward(self, x):
        pass

    


x = VQ(5, 5)
print(x.parameters())
optimizer = torch.optim.SGD(x.parameters(), lr=0.01, momentum=0.9)

for i in range(1000):
    values, indexes, loss = x(torch.tensor([[[1, 0, 1, 1, 0], [0, 1, 2, 3, 1]]], dtype=torch.float32))
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()