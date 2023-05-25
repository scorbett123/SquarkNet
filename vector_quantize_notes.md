## Encode

### Tranpose
BCT to BTC

### Calculate index
use:
dist = torch.sum(x\*\*2, dim=2, keepshape=True) + torch.sum(embeddings\*\*2, dim=1) - 2 * x . embeddings.t()
^ finds the euclidian distance between each vector to encode, and the embedding vectors
vals, indexes = torch.min(r, dim=2)

## Decode
one_hot = F.one_hot(indices)
result = torch.matmul(one_hot, embeddings)


## Calculate loss
loss is calculated using 
L = reconstruction loss + (euclidian distance from vector to encoded) + commitment loss * (euclidian distance from vector to encoded)
    trains encoder/decoder                  train embeddings                              trains encoder