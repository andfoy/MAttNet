uname -r
pwd
ls /input
ls /output

cd pyutils/mask-faster-rcnn
ls -l
./experiments/scripts/train_mask_rcnn_notime.sh 0 refcoco res101 notime
