#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3

python agent/table_main.py \
    --model_path /home/lym/data1/LLM-model/Qwen/Qwen2.5-32B-Instruct-AWQ \
    --load_mode vllm \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_max_model_len 16384 \
    --vllm_quantization AWQ \
    --batch_size 1 \
    --mapping_strategy llm \
    --data_dir /home/lym/PbCirculation/MarinePbFusion/outputs/pdf_outputs/qwen_output_2.1/4_reorganize_tables_data/reorganize_tables_csv \
    --output_file /home/lym/PbCirculation/MarinePbFusion/outputs/table_outputs/qwen_output_2.1/results.csv
