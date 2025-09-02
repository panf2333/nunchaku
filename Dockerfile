FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 

RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm -rf /root/miniconda3/miniconda.sh && \
    . /root/miniconda3/etc/profile.d/conda.sh && \
    conda init bash

ENV PATH="/root/miniconda3/bin:$PATH"

# Add this line to accept the Conda Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


RUN conda create -n image python=3.11 && \
    conda clean -afy

ENV PATH="/root/miniconda3/envs/image/bin:$PATH"

RUN pip install torch torchvision torchaudio && \
    pip install diffusers ninja wheel transformers accelerate sentencepiece protobuf && \
    pip install huggingface_hub peft opencv-python einops gradio spaces GPUtil && \
    ## https://github.com/nunchaku-tech/nunchaku/tree/main/app/flux.1/depth_canny
    pip install git+https://github.com/asomoza/image_gen_aux.git && \
    pip install controlnet_aux mediapipe && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 &&\ 
    conda install -c conda-forge gxx=11 gcc=11

ENV MAX_JOBS=8

COPY . /app

# Add this line with your GPU's compute capability
# https://developer.nvidia.com/cuda-gpus
ENV NUNCHAKU_INSTALL_MODE="ALL"

RUN git submodule init && \
    git submodule update && \
    python3 setup.py develop

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]