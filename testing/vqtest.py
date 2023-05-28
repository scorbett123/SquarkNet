import torch
import vq

time_length = 40
point_len = 64
batch_size = 64
quantizer = vq.RVQ(5, 1024, point_len)

# generate a random array of vectors of length x
points = torch.randn((batch_size, point_len, time_length))

vals, _ = quantizer(points)
print(torch.nn.functional.l1_loss(vals, points))

quantizer.initialise(points)

vals, _  = quantizer(points)
print(torch.nn.functional.l1_loss(vals, points))

