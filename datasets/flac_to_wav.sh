#!/bin/bash

folder=LibriSpeech/train-clean-360

for file in $(find "$folder" -type f -iname "*.flac")
do
    name=$(basename "$file" .flac)
    dir=$(dirname "$file")
    if [ ! -f "speech_train/$name.wav" ]; then
        ffmpeg -i "$file" "speech_train/$name".wav -hide_banner -loglevel error
        echo "$name"
    fi
    rm $file
done
