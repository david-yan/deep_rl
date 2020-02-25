FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         unzip \
         python3.7-dev \
         python3-distutils && \
     rm -rf /var/lib/apt/lists/*

# Link python3 to python
RUN ln -s /usr/bin/python3.7 /usr/bin/python

# Installs pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    pip install setuptools && \
    rm get-pip.py

ENV HOME /root
WORKDIR /root

# Install Python dependencies
COPY requirements.txt /root/requirements.txt
RUN pip install -r requirements.txt

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copy python files
COPY bc.py /root/bc.py
COPY load_policy.py /root/load_policy.py
COPY tf_util.py /root/tf_util.py

ENTRYPOINT ["python", "bc.py"]
