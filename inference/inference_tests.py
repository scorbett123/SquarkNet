import unittest
import inference
from model import models
import torch

class InferenceTests(unittest.TestCase):

    def test_aencode(self):
        model = models.Models.load("logs-t/epoch46/models.saved")
        input = "samples/epoch0/0-clean.wav"
        output = "test/a.sc"

        out = inference.wav_to_sc(input, output, model)
        self.assertTrue(out)

    def test_encodeshortlong(self):
        model = models.Models.load("logs-t/epoch46/models.saved")
        input = "samples/epoch0/0-clean.wav"
        output = "test/a.sc"

        long = inference.wav_to_sc(input, output, model)
        short = inference.wav_to_sc_short(input, output, model)
        
        same = torch.eq(long.data, short.data)
        self.assertTrue(torch.all(same), "All equal")

    def test_decode(self):
        model = models.Models.load("logs-t/epoch46/models.saved")
        input = "test/a.sc"
        output = "test/tmp.wav"

        out = inference.sc_to_wav(input, output, model)
        self.assertIsInstance(out, torch.Tensor)

    def test_decodeshortlong(self):
        model = models.Models.load("logs-t/epoch46/models.saved")
        output = "samples/epoch0/0-clean.wav"
        input = "test/a.sc"

        long = inference.sc_to_wav(input, output, model)
        short = inference.sc_to_wav_short(input, output, model)
        
        l1_diff = torch.nn.functional.l1_loss(long, short)
        self.assertAlmostEqual(l1_diff.item(), 0)

    def test_paddingsplit(self):
        d = [1,2,3,4,5,6,7,8,9,10]
        a = inference.split_with_padding(d, 1, 1)

        result = [[1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 10]]

        self.assertEqual(list(a), result)


if __name__ == "__main__":
    unittest.main()