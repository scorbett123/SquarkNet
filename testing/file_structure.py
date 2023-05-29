from operator import ior
import functools
from typing import Iterator

class File:
    MAGIC_NUM = 0x12321232  # TODO think of a more meaningful magic number, has to be 8 hex digits
    @staticmethod
    def read(filepath):
        reader = FileReader(filepath)
        num = reader.read_32_bit()
        print(hex(num))
        if num != File.MAGIC_NUM:  # check that the file is actually the correct type
            raise InvalidMagicNumerException
        
        length = reader.read_32_bit()
        n_codebooks = reader.read_byte()
        data_bit_depth = reader.read_byte()

        reader.read_byte()  # padding

        data = []
        for _ in range(length):
            sample = []
            for _ in range(n_codebooks):
                sample.append(reader.read_n_bits(n_bits=data_bit_depth))
            data.append(sample)
        return File(data, data_bit_depth=data_bit_depth, length=length, n_codebooks=n_codebooks)
        
    def __init__(self, data: Iterator[Iterator[int]], data_bit_depth: int, length=None, n_codebooks=None) -> None:
        self.length = length if length != None else len(data)
        self.n_codebooks = len(data[0])
        self.data_bit_depth = data_bit_depth
        self.data = data
        # TODO add exceptions for when data is of wrong length / wrong codebook count
    

    def write(self, filepath):
        writer = FileWriter(filepath)

        writer.write_32_bit(File.MAGIC_NUM)

        writer.write_32_bit(self.length)
        writer.write_byte(self.n_codebooks)
        writer.write_byte(self.data_bit_depth)

        writer.write_byte(0) # write a padding byte, not really needed, but makes analysis easier

        for sample in self.data:
            for codebook in sample:
                writer.write_n_bits(codebook, self.data_bit_depth)


        writer.close()


class InvalidMagicNumerException(Exception):
    pass

class EOFError(Exception):
    pass


class FileReader(): # A java DataInputStream inspired reader, you can probably see my java origins coming in through here, but we work on the level of bits, not bytes
    def __init__(self, path: str) -> None:
        self.file = open(path, mode="rb")
        self.bytes = [int.from_bytes([byte], "big") for byte in self.file.read()]
        self.front_pointer = 0 # The index of the next bit to be read

    def read_bit(self):
        if self.front_pointer // 8 == len(self.bytes):
            return EOFError
        index_in_byte = self.front_pointer % 8
        mask = 0b10000000 >> index_in_byte
        bit = (self.bytes[self.front_pointer // 8] & mask) >> (7-index_in_byte)
        self.front_pointer += 1
        return bit

    def read_byte(self):  # TODO create more efficient variant of this
        return self.read_n_bits(8)

    def read_n_bits(self, n_bits):
        bits = [self.read_bit() << (n_bits - 1 -i) for i in range(n_bits)] # create a list of bits shifted by their result position
        return functools.reduce(ior, bits) # bitwise or all of the bits to get the result
        
    def read_short(self):
        b1 = self.read_byte()
        b2 = self.read_byte()
        return (b1 << 8) + b2
    
    def read_32_bit(self):
        s1 = self.read_short()
        s2 = self.read_short()
        return (s1 << 16) + s2
    
    def close(self):
        self.file.close()


class FileWriter():
    def __init__(self, path: str) -> None:
        self.file = open(path, mode="wb")
        self.bytes = []
        self.front_pointer = -1

    def write_bit(self, bit):
        bit &= 1 # just in case everything else has gone wrong
        self.front_pointer += 1
        if len(self.bytes) <= (self.front_pointer // 8):  # we need to be careful here incase someone has messed with self.front_pointer, and it hasn't necesarily incremented how we'd expect
            self.bytes.append(0)

        index_in_byte = self.front_pointer % 8
        self.bytes[self.front_pointer // 8] |= (bit << (7-index_in_byte))  # use or here to incase there is any existing data

    def write_byte(self, byte):  # TODO, this should interact directly with bytes, but need to take into account the byte may be split
        self.write_n_bits(byte, n_bits=8)
        
    def write_short(self, short):
        self.write_byte(short >> 8)
        self.write_byte(short & 0xFF)
    
    def write_32_bit(self, integer):  # java typing is so much better, not really sure how to handle signing, so for now, we don't
        self.write_short(integer >> 16)
        self.write_short(integer & 0xFFFF)

    def write_n_bits(self, to_write, n_bits):
        for i in range(n_bits):
            self.write_bit(to_write >> (n_bits - 1 - i))

    def close(self):
        self.file.write(bytes(byte & 0xFF for byte in self.bytes))
        self.file.close()
        

        
# just some testing stuff
if __name__ == "__main__": 
    f = File(data=[[10, 10, 10, 10, 10]], data_bit_depth=5)
    f.write("test.test")

    x = File.read("test.test")
    print(x.data)