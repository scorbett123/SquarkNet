## Encode

### Tranpose
BCT to BTC

### Calculate index
use:
dist = torch.sum(x\*\*2, dim=2, keepshape=True) + torch.sum(embeddings\*\*2, dim=1) - 2 * x . embeddings.t()
vals, indexes = torch.min(r, dim=2)

## Decode
one_hot = F.one_hot(indices)
result = torch.matmul(one_hot, embeddings)




## Calculate loss