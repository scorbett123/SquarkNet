import torch
import vq
import tqdm
import unittest
from torch.utils.tensorboard import SummaryWriter

# point_len = 5
# quantizer = vq.VQ(1024, point_len).to("cuda")
# with torch.no_grad():
#     p = []
#     y = []
#     # generate a random array of vectors of length x
#     for i in tqdm.tqdm(range(10000)):
#         points = torch.normal(0,20,(20000, point_len)).to("cuda")
#         y.append(quantizer(points)[2])
#         quantizer.frozen_kmeans(points, torch.Tensor([False] *1024).to(torch.bool),kmeans_iters=1)
#         x = quantizer(points)[2]
#         p.append(x)
#     print(sum(p) / len(p))
#     print(sum(y) / len(y))

residual = False
x = vq.RVQ(5, 5, 5) if residual else vq.VQ(5, 5)
optimizer = torch.optim.SGD(x.parameters(), lr=0.01, momentum=0.9)
writer = SummaryWriter(log_dir="logs_test/")
data = torch.normal(mean=2, std=2, size=(200, 5))
for i in tqdm.tqdm(range(1000)):
    values, indexes, loss = x(data)
    # print(loss)
    writer.add_scalar("loss", loss, i)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

exit()

class MovingAverageTests(unittest.TestCase):

    def test_basic_vq_cache(self):
        cache = vq.VectorCache(3)

        cache.add_vector(torch.Tensor([1]))
        self.assertTrue(torch.equal(cache.concat(), torch.Tensor([1])))
        cache.add_vector(torch.Tensor([2]))
        self.assertTrue(torch.equal(cache.concat(), torch.Tensor([1, 2])))

    def test_expiry_vq_cahce(self):
        cache = vq.VectorCache(3)
        for i in range(50):
            cache.add_vector(torch.Tensor([i]))

        self.assertTrue(torch.equal(cache.concat(), torch.Tensor([47, 48, 49])))

if __name__ == "__main__":
    unittest.main()