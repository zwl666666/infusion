#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=2 python scripts/infusion_txt2img.py --ddim_eta 0.0 \
                                    --outdir ./outputs/cat  \
                                    --seed 2023 \
                                    --steps 50  \
                                    --scale 8.0 \
                                    --beta 0.7 \
                                    --tau 0.15 \
                                    --n_samples 10 \
                                    --n_iter 1 \
                                    --personalized_ckpt ckpt/2023-12-16T17-47-12_cat/models/step=1500.ckpt \
                                    --target_word "cat" \
                                    --prompt "A cat is strolling through a bustling cityscape"


