uname -r
pwd
ls /input
ls /output

cd pyutils/mask-faster-rcnn
ls -l
cd lib
make
cd ..
bash experiments/scripts/train_mask_rcnn_notime.sh 1 endovis_2017 res101 notime
