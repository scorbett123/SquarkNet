import unittest
import file_structure
import random

class FileTests(unittest.TestCase):
    def test_bit(self):
        writer = file_structure.FileWriter("/tmp/x.test")
        order = [random.randrange(0, 2) for i in range(1000)]

        for i in order:
            writer.write_bit(i)

        writer.close()

        reader = file_structure.FileReader("/tmp/x.test")

        for i in order:
            self.assertEqual(reader.read_bit(), i)
        reader.close()
        

    def test_byte(self):
        writer = file_structure.FileWriter("/tmp/x.test")
        order = [random.randrange(0, 256) for i in range(1000)]

        for i in order:
            writer.write_byte(i)

        writer.close()

        reader = file_structure.FileReader("/tmp/x.test")

        for i in order:
            self.assertEqual(reader.read_byte(), i)
        reader.close()

    
    def test_nbit(self):
        writer = file_structure.FileWriter("/tmp/Y.test")
        order = [(bit_depth, random.randrange(0, 2 ** (bit_depth - 1))) for bit_depth in [random.randrange(1, 50) for i in range(100000)]]  # don't particularly like the random here, but enough iterations mean likelyhood of false working is very low

        for i in order:
            writer.write_n_bits(i[1], i[0])

        writer.close()

        reader = file_structure.FileReader("/tmp/Y.test")

        for i in order:
            self.assertEqual(reader.read_n_bits(i[0]), i[1])
        reader.close()



if __name__ == "__main__":
    unittest.main()