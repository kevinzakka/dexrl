#!/bin/bash

set -ex

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=7 python train.py --use_wandb
