
FROM alpine:3.19 AS weights
RUN apk add --no-cache curl && \
    mkdir -p /weights/config && \
    curl -L -o /weights/config/GroundingDINO_SwinT_OGC.py \
        https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/refs/heads/main/groundingdino/config/GroundingDINO_SwinT_OGC.py && \
    curl -L -o /weights/groundingdino_swint_ogc.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0+PTX 8.6+PTX 8.9+PTX"

# Install system dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    git \
    python3-opencv \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    procps && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /processor

RUN git clone https://github.com/signifyai/grounding-dino.git
ENV PATH=/usr/local/cuda/bin:$PATH

# add comment for build

# Install Poetry
ENV POETRY_VERSION=2.1.3
RUN pip install poetry==${POETRY_VERSION}

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies with Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root --only main

RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

RUN cd grounding-dino/ && python -m pip install . --no-cache-dir --no-deps --no-build-isolation

COPY --from=weights /weights /weights

COPY . .

ENV PORT=8080

CMD ["python", "-m", "app.main"]