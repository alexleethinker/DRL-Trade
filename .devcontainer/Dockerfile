FROM python:3.10

# ## install finrl library
RUN apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig && \
    apt-get -y autoclean && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/* 
RUN pip --no-cache-dir install --upgrade wrds swig
RUN pip --no-cache-dir install git+https://github.com/AI4Finance-Foundation/FinRL.git
