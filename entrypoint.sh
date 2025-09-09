#!/bin/bash

# Activate the Conda environment
# The source command is necessary to apply the environment changes to the current shell.
source /opt/conda/bin/activate image

# huggingface-cli login
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    hf auth login --token $HUGGINGFACE_TOKEN
fi


exec "$@"
