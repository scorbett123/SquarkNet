from scipy.io import wavfile
from pesq import pesq
import glob
import os
from tqdm import tqdm
import random
from encodec import EncodecModel
from encodec.utils import convert_audio
import torch

model = EncodecModel.encodec_model_24khz().to("cuda")
model.set_target_bandwidth(24)

audio_files = random.sample(glob.glob("datasets/speech_train/*.wav"), 500)
values = []


os.system(f'ffmpeg -to 50 -f lavfi -i "aevalsrc=random(4)" -y -acodec pcm_s16le -ac 1 -ar 16000  ./evaluation/tmp/tmp.wav')
for file in tqdm(audio_files):
    rate, ref = wavfile.read(file)

    # we encode
    #os.system(f"ffmpeg -i {file} -y -c:a libopus -b:a 1.5k ./evaluation/tmp/tmp.opus >/dev/null 2>&1")

    
    # x = convert_audio(ref, rate, model.sample_rate, 1, device="cuda")
    # x = x.unsqueeze(0)

    # with torch.no_grad():
    #     encoded = model.encode(x)
    #     decoded = model.decode(encoded)

    # y = convert_audio(decoded, model.sample_rate, rate, 1, device="cuda").squeeze(0)

    # os.system(f"../lyra/lyra/bazel-bin/lyra/cli_example/encoder_main --input_path={file} --output_dir=evaluation/tmp/ --bitrate=12000 --model_path ../lyra/lyra/lyra/model_coeffs >/dev/null 2>&1")
    # os.system(f"../lyra/lyra/bazel-bin/lyra/cli_example/decoder_main --encoded_path=evaluation/tmp/{file.split('.')[0].split('/')[-1]}.lyra --output_dir=evaluation/tmp/ --bitrate=12000 --model_path ../lyra/lyra/lyra/model_coeffs >/dev/null 2>&1")
    #os.system(f"ffmpeg -i ./evaluation/tmp/tmp.opus -y -acodec pcm_s16le -ac 1 -ar 16000 ./evaluation/tmp/tmp.wav >/dev/null 2>&1")


    rate2, res = wavfile.read(f"./evaluation/tmp/tmp.wav")
    res = res[:ref.shape[0]]
    assert rate == rate2
    values.append(pesq(rate, ref, res, 'wb'))

    # os.remove(f"evaluation/tmp/{file.split('.')[0].split('/')[-1]}.lyra")
    # os.remove(f"evaluation/tmp/{file.split('.')[0].split('/')[-1]}_decoded.wav")

print(sum(values) / len(values))