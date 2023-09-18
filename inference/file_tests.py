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
        

    def test_byte(self):
        writer = file_structure.FileWriter("/tmp/x.test")
        order = [random.randrange(0, 256) for i in range(1000)]

        for i in order:
            writer.write_byte(i)

        writer.close()

        reader = file_structure.FileReader("/tmp/x.test")

        for i in order:
            self.assertEqual(reader.read_byte(), i)

    
    def test_nbit(self):
        writer = file_structure.FileWriter("/tmp/Y.test")
        order = [(bit_depth, random.randrange(0, 2 ** bit_depth)) for bit_depth in [random.randrange(1, 10) for i in range(10)]]
        order = [(7, 140), (7, 80)]
        for i in order:
            print(i)
            writer.write_n_bits(i[1], i[0])

        writer.close()

        reader = file_structure.FileReader("/tmp/Y.test")
        print("onto reading")

        for i in order:
            print(i)
            self.assertEqual(reader.read_n_bits(i[0]), i[1])


if __name__ == "__main__":
    unittest.main()