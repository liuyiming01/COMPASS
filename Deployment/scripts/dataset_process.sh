#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3

python agent/dataset_processor/main.py \
    --model_path /home/lym/data1/LLM-model/Qwen/Qwen2.5-32B-Instruct \
    --dataset_dir /home/lym/PbCirculation/Data/csv_file \
    --output_dir outputs \