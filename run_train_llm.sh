#!/bin/bash
# conda activate your_env
export CUDA_LAUNCH_BLOCKING=1
deepspeed --num_gpus=2 train_llm.py