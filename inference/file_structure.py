from operator import ior
import functools
from typing import Iterator
import timeit
import math

class File:
    MAGIC_NUM = 0x12121212  # TODO think of a more meaningful magic number, has to be 8 hex digits
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
        reader.close()
        return File(data, data_bit_depth=data_bit_depth, length=length, n_codebooks=n_codebooks)
        
    def __init__(self, data: Iterator[Iterator[int]], data_bit_depth: int, length=None, n_codebooks=None) -> None:
        self.length = length if length != None else len(data)
        self.n_codebooks = n_codebooks if n_codebooks != None else len(data[0])
        self.data_bit_depth = data_bit_depth
        self.data = data

        self.file_length = math.ceil(self.length * self.n_codebooks * self.data_bit_depth / 8) + 11
        # TODO add exceptions for when data is of wrong length / wrong codebook count
    

    def write(self, filepath):
        writer = FileWriter(filepath, self.file_length)

        writer.write_32_bit(File.MAGIC_NUM)

        writer.write_32_bit(self.length)
        writer.write_byte(self.n_codebooks)
        writer.write_byte(self.data_bit_depth)

        writer.write_byte(0) # write a padding byte, not really needed, but makes analysis easier

        x = 0
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
        if (self.front_pointer + n_bits - 1) // 8 == len(self.bytes):
            raise EOFError
        f_in_byte = (self.front_pointer) % 8

        if f_in_byte == 0 and n_bits >= 8:  # this doesn't always help, but can sometimes bring around great benefits
            val = self.bytes[self.front_pointer // 8]
            self.front_pointer += 8
            if n_bits > 8:
                return val << (n_bits - 8) | self.read_n_bits(n_bits-8)
            else:
                return val << (n_bits - 8)

        fit = 8-f_in_byte

        space_after = max(fit-n_bits, 0)
        space_before = f_in_byte + max(n_bits - fit, 0)

        m1 = (1 << fit) - 1  # eliminate before
        m2 = ~((1 << space_after) - 1)  # eliminate after
        mask = m1 & m2

        val = ((self.bytes[self.front_pointer // 8] & mask) >> space_after) << (space_before)
        self.front_pointer += min(fit, n_bits)
        remaining = n_bits - fit
        if remaining > 0:
            return val + self.read_n_bits(remaining)
        else:
            return val

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
    def __init__(self, path: str, byte_num=0) -> None:
        self.file = open(path, mode="wb")
        self.bytes = [0] * byte_num
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

    def write_n_bits(self, to_write, n_bits):  # this looks like a pain, and can probably be further improved in the future, but can't deny 2x speed improvement
        # print(self.front_pointer, n_bits)
        f_in_byte = (self.front_pointer + 1) % 8

        if f_in_byte == 0 and n_bits >= 8:  # the benefit of this changes dependant on bitdpeth, ranging from 3x improvement, to none
            self.bytes[(self.front_pointer+1) // 8] = to_write >> (n_bits - 8)
            self.front_pointer += 8

            if n_bits > 8:
                self.write_n_bits(to_write, n_bits-8)
            return

        fit = 8 - f_in_byte
        space_after = max(fit - n_bits, 0)  # how much space do we have after
        space_before = f_in_byte + max(n_bits - fit, 0)

        m1 = (1 << fit) - 1  # eliminate before
        m2 = ~((1 << space_after) - 1)  # eliminate after

        mask = m1 & m2
        result = ((to_write << space_after) >> space_before) & mask

        # if len(self.bytes) <= ((self.front_pointer+1) // 8):  # we need to be careful here incase someone has messed with self.front_pointer, and it hasn't necesarily incremented how we'd expect
        #     self.bytes.append(0)
        
        self.bytes[(self.front_pointer+1) // 8] |= result

        self.front_pointer += min(fit, n_bits)

        remaining = n_bits - fit
        if remaining > 0:
            self.write_n_bits(to_write, remaining)


        # for i in range(n_bits):
        #     self.write_bit(to_write >> (n_bits - 1 - i))

    def close(self):
        self.file.write(bytes(byte & 0xFF for byte in self.bytes))
        self.file.close()
        

        
# just some testing stuff
if __name__ == "__main__": 
    s = timeit.default_timer()
    f = File(data=[[10, 10, 10, 10, 10] for i in range(50000)], data_bit_depth=8)
    f.write("test.test")
    x = timeit.default_timer()
    # print("Writing", x-s)
    # exit()

    f = File.read("test.test")
    y = timeit.default_timer()
    print("Writing", x-s)
    print("reading", y-x)

    
    print(len(f.data))
    for i in f.data:
        for j in i:
            assert j == 10

    # writer = FileWriter("test.test")
    # for i in range(100):
    #     writer.write_n_bits(10, 10)
    # writer.close()

    # reader = FileReader("test.test")
    # for i in range(10):
    #     print(reader.read_n_bits(10))