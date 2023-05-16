#!/bin/bash

folder=/home/sam/ml/nea/datasets/dev-other

for file in $(find "$folder" -type f -iname "*.flac")
do
    name=$(basename "$file" .flac)
    dir=$(dirname "$file")
    ffmpeg -i "$file" "speech/$name".wav
    #ffmpeg -i $file $dir/$name.wav
    #rm $file
done