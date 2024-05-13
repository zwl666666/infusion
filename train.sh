#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --name backpack_new \
    --base ./configs/kv_train/backpack.yaml \
    --basedir ./kv_ckpt \
    -t True \
    --gpus 0,

CUDA_VISIBLE_DEVICES=1 python main.py \
    --name cat_new \
    --base ./configs/kv_train/cat.yaml \
    --basedir ./kv_ckpt \
    -t True \
    --gpus 0,

CUDA_VISIBLE_DEVICES=1 python main.py \
    --name dog5_new \
    --base ./configs/dog5.yaml \
    --basedir ./kv_ckpt \
    -t True \
    --gpus 0,

CUDA_VISIBLE_DEVICES=1 python main.py \
    --name pot_new \
    --base ./configs/kv_train/pot.yaml \
    --basedir ./kv_ckpt \
    -t True \
    --gpus 0,

CUDA_VISIBLE_DEVICES=1 python main.py \
    --name teddy_new \
    --base ./configs/kv_train/teddy_bear.yaml \
    --basedir ./kv_ckpt \
    -t True \
    --gpus 0,