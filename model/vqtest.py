import torch
import vq
import tqdm

point_len = 5
quantizer = vq.VQ(1024, point_len).to("cuda")
with torch.no_grad():
    p = []
    y = []
    # generate a random array of vectors of length x
    for i in tqdm.tqdm(range(100)):
        points = torch.normal(1,3,(20000, point_len)).to("cuda")
        y.append(quantizer(points)[2])
        quantizer.frozen_kmeans(points, torch.Tensor([False] *1024).to(torch.bool),kmeans_iters=1)
        x = quantizer(points)[2]
        p.append(x)
    print(sum(p) / len(p))
    print(sum(y) / len(y))



