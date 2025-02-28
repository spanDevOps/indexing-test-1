FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    unzip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

# Add conda to path and initialize
SHELL ["/bin/bash", "-c"]
RUN $CONDA_DIR/bin/conda init bash && \
    echo "conda activate cuda118" >> ~/.bashrc

# Create conda environment with Python 3.9
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda create -n cuda118 python=3.9.21 -y

# Install conda packages
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate cuda118 && \
    conda install -y -c conda-forge \
    cudatoolkit=11.8.0 \
    numpy=1.26.4 \
    mkl-service=2.4.0 \
    faiss-gpu=1.7.2 \
    six=1.17.0 \
    && conda clean -afy

# Install PyTorch
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate cuda118 && \
    conda install -y -c pytorch -c nvidia \
    pytorch=2.1.0 \
    pytorch-cuda=11.8 \
    torchvision=0.16.0 \
    && conda clean -afy

# Install pip packages
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate cuda118 && \
    pip install --no-cache-dir \
        albumentations==1.3.1 \
        annotated-types==0.7.0 \
        boto3==1.35.91 \
        botocore==1.35.91 \
        click==8.1.8 \
        coloredlogs==15.0.1 \
        contourpy==1.3.0 \
        dnspython==2.7.0 \
        easydict==1.10 \
        exceptiongroup==1.2.2 \
        filelock==3.13.1 \
        fsspec==2024.2.0 \
        h11==0.14.0 \
        humanfriendly==10.0 \
        importlib-resources==6.4.5 \
        insightface==0.7.3 \
        jmespath==1.0.1 \
        joblib==1.4.2 \
        kiwisolver==1.4.7 \
        matplotlib==3.7.1 \
        onnx==1.17.0 \
        onnxruntime-gpu==1.16.3 \
        opencv-python-headless==4.10.0.84 \
        pillow==9.5.0 \
        prettytable==3.7.0 \
        protobuf==5.29.2 \
        psutil==6.1.1 \
        pydantic==2.10.4 \
        pydantic-core==2.27.2 \
        pymongo==4.10.1 \
        pyparsing==3.2.1 \
        python-dateutil==2.9.0.post0 \
        python-dotenv==1.0.1 \
        pywavelets==1.6.0 \
        qudida==0.0.4 \
        scikit-image==0.19.3 \
        scikit-learn==1.2.2 \
        scipy==1.10.1 \
        sniffio==1.3.1 \
        starlette==0.41.3 \
        sympy==1.13.1 \
        threadpoolctl==3.5.0 \
        tifffile==2024.8.30 \
        urllib3==1.26.20 \
        uvicorn==0.34.0 \
        watchtower==3.3.1 \
        wcwidth==0.2.13 \
        zipp==3.21.0

# Set the default conda environment
ENV CONDA_DEFAULT_ENV=cuda118
ENV CONDA_PREFIX=$CONDA_DIR/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Create all necessary directories
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    mkdir -p /app/src/models && \
    mkdir -p /tmp/face_processing

# Copy application code
COPY . /app/

# Set permissions
RUN chmod -R 755 /root/.insightface && \
    chmod -R 755 /app/src/models && \
    chmod -R 777 /tmp/face_processing

# Download InsightFace models
RUN cd /root/.insightface/models/buffalo_l && \
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip buffalo_l.zip && \
    rm buffalo_l.zip

WORKDIR /app

# Add and configure startup script
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Use bash as shell and source bashrc in entrypoint
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", ". ~/.bashrc && /app/startup.sh"]