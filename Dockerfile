# This is a potassium-standard dockerfile, compatible with Banana

# Must use a Cuda version 11+
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

WORKDIR /

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git awscli gcc g++ make cmake

# Install python packages
ADD requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install detectron2==0.1.3 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html \
    && pip install -r requirements.txt

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

# Add your model weight files 
RUN aws s3 cp s3://resource-bucket-iengineering-dev/cb_detection_weight.pth ./model_weights/model.pth
# (in this case we have a python script)
# ADD download.py .
# RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py