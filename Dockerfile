# ================================
# 베이스 이미지: CUDA 11.6 + Ubuntu 20.04
# ================================
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Install system dependencies and python3.9 dev headers
RUN apt-get clean && apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8 python3.8-dev python3-pip git wget unzip nano \
    libglib2.0-0 libsm6 libxext6 libxrender-dev build-essential && \
    ln -sf /usr/bin/python3.8 /usr/bin/python 

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch (CUDA 11.6 build)
RUN python -m pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# Install Opencv-python
RUN pip install opencv-python 


WORKDIR /workspace

# 필요하다면 detectron2나 추가 패키지 설치
# 예시:
# RUN git clone https://github.com/facebookresearch/detectron2.git && \
#     cd detectron2 && \
#     pip install -e .

# 컨테이너 진입 시 bash 실행
CMD ["/bin/bash"]
