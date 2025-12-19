#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --name cat \
    --base ./configs/infusion_cat.yaml \
    --basedir ./kv_ckpt \
    -t True \
    --gpus 0,
