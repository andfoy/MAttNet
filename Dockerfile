FROM pytorch/pytorch:0.4-cuda9-cudnn7-devel
RUN conda install cython -y
RUN pip install pycocotools
# RUN conda install cython -y

COPY . /mattnet
WORKDIR /mattnet/pyutils/mask-faster-rcnn/lib
RUN make

WORKDIR /mattnet/pyutils/mask-faster-rcnn/lib/layer_utils/roi_pooling
# WORKDIR /mattnet/pyutils/mask-faster-rcnn/lib/nms
RUN python -c "from roi_pool import RoIPoolFunction"
WORKDIR /mattnet/pyutils/mask-faster-rcnn/lib/nms
RUN python -c "from pth_nms import pth_nms"

# CMD "uname -r"
# CMD "nvidia-smi" 
#CMD "ls /input && cd /output && touch some_file"
WORKDIR /mattnet/
CMD ["bash", "run.sh"]
 
