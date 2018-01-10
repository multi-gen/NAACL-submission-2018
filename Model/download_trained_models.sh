#!/bin/bash
# Downloads and uncompresses our trained models for both Arabic and Esperanto.
wget -O Checkpoints.zip https://www.dropbox.com/s/w227sh3x0zuevug/Checkpoints.zip?dl=1
unzip -o Checkpoints.zip
rm Checkpoints.zip

# Downloads and uncompresses all the files that are required for sampling from our trained models.
wget -O data.zip https://www.dropbox.com/s/yxnddto45k0v3bj/data.zip?dl=1
unzip -o data.zip
rm data.zip
echo All required files have been downloaded and un-compressed successfully.
