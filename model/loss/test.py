import unittest
from model.loss.moving_average import *

class MovingAverageTests(unittest.TestCase):

    def test_simple(self):
        simple_average = SMA(100)
        self.assertEqual(simple_average.average, 1, "Initial average should be 1")

        self.assertEqual(simple_average.update(100), 1.99)
        self.assertEqual(simple_average.update(100), 2.98)

        for _ in range(100):
            simple_average.update(100)
        
        self.assertEqual(simple_average.average, 100)

    def test_exponential(self):
        exponental_average = EMA(100)
        self.assertEqual(exponental_average.average, 1, "Initial average should be 1")
        self.assertAlmostEqual(exponental_average.update(100).item(), 20.08, places=2)
        self.assertAlmostEqual(exponental_average.update(100).item(), 29.63, places=2)

        for _ in range(100):
            exponental_average.update(100)
        
        self.assertEqual(exponental_average.average, 100)

    def test_initial_value(self):
        self.assertEqual(EMA(10, intial_val=10).average, 10)
        sma = SMA(10, intial_val=10)
        self.assertTrue(torch.all(sma._window == 10))  # check that the window is all initialised correctly

        


if __name__ == "__main__":
    unittest.main()