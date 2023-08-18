FROM python:3.10


# ## install finrl library
RUN apt-get update -y -qq && apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig
RUN pip install wrds swig
RUN pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

WORKDIR /src