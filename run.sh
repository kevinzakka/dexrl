#!/bin/bash

set -ex

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 python train.py --num_episodes 10000000 --start_training 100 --use_wandb
