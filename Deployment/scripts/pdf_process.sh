#!/bin/bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3

python agent/pdf_main.py \
    --steps reorganize_tables_data \
    --model_path /home/lym/data1/LLM-model/Qwen/Qwen2.5-32B-Instruct-AWQ \
    --load_mode vllm \
    --vllm_gpu_memory_utilization 0.75 \
    --vllm_max_model_len 20000 \
    --vllm_quantization AWQ \
    --prompts_config /home/lym/PbCirculation/COMPASS/Deployment/agent/config/pdf_config/qwen_prompts2.2.yaml \
    --batch_size 1 \
    --input_csv /home/lym/PbCirculation/COMPASS/Deployment/outputs/pdf_outputs/qwen_output_2.1/3_extract_pb_tables/results.csv \
    --pdf_dir /home/lym/PbCirculation/MarinePbFusion/data/pdfs/all_acemap_Pb/marine_pdfs_20250419_162820 \
    --output_dir /home/lym/PbCirculation/COMPASS/Deployment/outputs/pdf_outputs/qwen_output_2.1
