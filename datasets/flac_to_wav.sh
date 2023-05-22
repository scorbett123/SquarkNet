#!/bin/bash

folder=LibriSpeech/dev-clean

for file in $(find "$folder" -type f -iname "*.flac")
do
    name=$(basename "$file" .flac)
    dir=$(dirname "$file")
    ffmpeg -i "$file" "speech/$name".wav -hide_banner -loglevel error
    echo "$name"
    #ffmpeg -i $file $dir/$name.wav
    #rm $file
done
