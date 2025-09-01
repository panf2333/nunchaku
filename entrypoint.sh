#!/bin/bash

source ~/miniconda3/bin/activate image

# huggingface-cli login
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    hf auth login --token $HUGGINGFACE_TOKEN
fi


exec "$@"