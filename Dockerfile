FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel
RUN conda install cython -y
RUN pip install pycocotools
RUN pip install h5py numpy matplotlib \
scipy \
scikit-image \
easydict==1.6 \
opencv-python \
jupyter \
tensorboardX
# RUN conda install cython -y
RUN apt-get update
RUN apt-get -y install libgtk2.0-dev libglib2.0-0

COPY . /mattnet
WORKDIR /mattnet/pyutils/mask-faster-rcnn/lib
RUN make

# WORKDIR /mattnet/pyutils/mask-faster-rcnn/lib/layer_utils/roi_pooling
# # WORKDIR /mattnet/pyutils/mask-faster-rcnn/lib/nms
# RUN python -c "from roi_pool import RoIPoolFunction"
# WORKDIR /mattnet/pyutils/mask-faster-rcnn/lib/nms
# RUN python -c "from pth_nms import pth_nms"

# CMD "uname -r"
# CMD "nvidia-smi"
#CMD "ls /input && cd /output && touch some_file"
WORKDIR /mattnet/
CMD ["bash", "run.sh"]
