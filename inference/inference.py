from model.datasets import *
import torch
from model import models
import torch
import file_structure
from model.utils import norm
import torchaudio
import traceback
from typing import Callable


def split_with_padding(data, split_len: int, padding_len: int):
    current = 0
    while current < len(data):
        yield data[max(current - padding_len, 0): current + split_len + padding_len]
        current += split_len + padding_len



@torch.no_grad()
def sc_to_wav(path, output_path, model: models.Models, progress_callback: Callable[[float], None] = None):
    try:
        if not output_path.endswith(".wav"):
            output_path += ".wav"

        f = file_structure.File.read(path)
        
        indices = torch.tensor(f.data).unsqueeze(0)
        
        out = model.decode(indices).squeeze(0)
        torchaudio.save(output_path, out.cpu(), sample_rate=16000)

        
        progress_callback(1)
        return True
    except Exception:
        traceback.print_exc()
        return False  # we have failed
    

@torch.no_grad()
def wav_to_sc(path, output_path, model: models.Models, progress_callback: Callable[[float], None] = None):
    try:
        if not output_path.endswith(".sc"):
            output_path += ".sc"
        sound, sample_rate = torchaudio.load(path)
        audio_data = norm(sound)

        if sample_rate != 16000:
            audio_data = torchaudio.functional.resample(sound, orig_freq=sample_rate, new_freq=16000)

        codebooks = None
        padding_len = 10
        segment_len = 290

        segment_count = math.ceil( audio_data.shape[1] / ((padding_len + segment_len) * model.ctx_len))
        for i, segment in enumerate(split_with_padding(audio_data.squeeze(0), segment_len * model.ctx_len, padding_len * model.ctx_len)):
            segment_books = model.encode(segment.unsqueeze(0)).squeeze(0)
            if codebooks == None:
                codebooks = segment_books[:segment_len+int(padding_len / 2)]
            else:
                without_padding = segment_books[int(padding_len / 2) : segment_len+padding_len+int(padding_len / 2)]
                codebooks = torch.concat((codebooks, without_padding), dim=0)
            
            if progress_callback != None:
                progress_callback(i / segment_count)

        f = file_structure.File(codebooks, data_bit_depth=math.ceil(math.log2(model.ncodes)), n_codebooks=model.nbooks)
        f.write(output_path)
        progress_callback(1)
        return f
    except Exception:  # absolutely any exception in here and we just return failed
        traceback.print_exc()
        return False


@torch.no_grad()
def wav_to_sc_short(path, output_path, model: models.Models):
    """ This does no clever stuff to reduce memory usage, ONLY intended as a test for if result is the same """
    try:
        if not output_path.endswith(".sc"):
            output_path += ".sc"
        sound, sample_rate = torchaudio.load(path)
        sound = norm(sound)

        if sample_rate != 16000:
            sound = torchaudio.functional.resample(sound, orig_freq=sample_rate, new_freq=16000)

        audio_data = sound.unsqueeze(0)
        codebooks = model.encode(audio_data)

        f = file_structure.File(codebooks.squeeze(0), data_bit_depth=math.ceil(math.log2(model.ncodes)), n_codebooks=model.nbooks)
        f.write(output_path)
        return f
    except Exception:  # absolutely any exception in here and we just return failed
        traceback.print_exc()
        return False
