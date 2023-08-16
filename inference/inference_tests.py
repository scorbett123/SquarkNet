import unittest
import inference
from model import models

class InferenceTests(unittest.TestCase):

    def test_aencode(self):
        model = models.Models.load("logs-t/epoch46/models.saved")
        input = "samples/epoch0/0-clean.wav"
        output = "test/a.sc"

        out = inference.wav_to_sc(input, output, model)
        self.assertTrue(out)

    def test_decode(self):
        model = models.Models.load("logs-t/epoch46/models.saved")
        input = "test/a.sc"
        output = "test/tmp.wav"

        out = inference.sc_to_wav(input, output, model)
        self.assertTrue(out)

        


if __name__ == "__main__":
    unittest.main()