

GPU_ID=$1
DATASET=$2
SPLITBY=$3

IMDB="endovis_2017_group2"
ITERS=27600
TAG="notime"
NET="res101"
ID="mrcn_cmr_with_st"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ./tools/train.py \
    --imdb_name ${IMDB} \
    --net_name ${NET} \
    --iters ${ITERS} \
    --tag ${TAG} \
    --dataset ${DATASET} \
    --splitBy ${SPLITBY} \
    --max_iters 30000 \
    --with_st 1 \
    --id ${ID} >> /output/train_group2.log 2>&1
