ARG PYTORCH="1.13.1"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN rm /etc/apt/sources.list.d/cuda.list \
#     && rm /etc/apt/sources.list.d/nvidia-ml.list \
#     && apt-key del 7fa2af80 \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# nvim dev env
#	RUN apt-get update \ 
#     && apt-get install -y wget python3-venv ca-certificates curl gnupg unzip \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# nvim dev env
# RUN mkdir -p /etc/apt/keyrings \
#     && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
#     && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_16.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
#     && apt-get update \
#     && apt-get install nodejs -y

# nvim dev env
# RUN apt update \
#     && apt upgrade -y
    
# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine==0.7.1" "mmcv==2.0.0rc4"

# Install EasyOCR
RUN pip install easyocr

RUN pip install "numpy<2"

# Install MMDetection
RUN conda clean --all \
    && git clone -b v3.1.0 https://github.com/open-mmlab/mmdetection.git /mmdetection \
    && cd /mmdetection \
    && pip install --no-cache-dir -e .

# nvim dev env
# RUN mkdir -p /opt/nvim/bin \
#     && touch /opt/nvim/bin/nvim
# RUN update-alternatives --install /usr/bin/nvim nvim /opt/nvim/bin/nvim 091

# tensorboard
# RUN pip install future tensorboard

WORKDIR /project
