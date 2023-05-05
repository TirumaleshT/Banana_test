# This is a potassium-standard dockerfile, compatible with Banana

# Must use a Cuda version 11+
FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

WORKDIR /

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git awscli gcc g++ make cmake \
	libgl1-mesa-glx ffmpeg libsm6 libxext6

# Install python packages
ADD requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html \
    && pip install -r requirements.txt

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

# Add your model weight files 
RUN aws s3 cp s3://resource-bucket-iengineering-dev/cb_detection_weight.pth ./model_weights/model.pth
# (in this case we have a python script)
# ADD download.py .
# RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py