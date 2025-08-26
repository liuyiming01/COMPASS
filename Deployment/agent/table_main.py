# table_main.py
import argparse
import torch
from pathlib import Path

from pdf_processor.llm_loader import LLMHandler
from table_processor.utils import setup_logging
from table_processor.pipeline import TablePipeline

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Standardize lead (Pb) data from oceanographic CSV files')

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--load_mode", type=str, default="transformers", choices=["transformers", "vllm"])
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--vllm_max_model_len", type=int, default=16384)
    parser.add_argument("--vllm_quantization", type=str, default="AWQ")

    parser.add_argument('--data_dir', type=str, required=True)
    project_root = Path(__file__).parent.parent
    default_output = project_root / "outputs" / "table_outputs" / "qwen_output_2.0" / "standardized_pb_data.csv"
    parser.add_argument('--output_file', type=str, default=str(default_output))
    parser.add_argument("--mapping_strategy", type=str, default="llm", choices=["llm", "rule_based"],)

    return parser.parse_args()

def main():
    args = parse_arguments()
    setup_logging()

    llm_handler = LLMHandler(
        args.model_path,
        load_mode=args.load_mode,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        vllm_quantization=args.vllm_quantization
    )
    generator = llm_handler.create_generator(max_new_tokens=512)

    pipeline = TablePipeline(strategy=args.mapping_strategy)
    pipeline.process(
        input_dir=args.data_dir,
        output_path=args.output_file,
        llm_generator=generator,
        batch_size=args.batch_size,
    )

    if args.load_mode == "vllm":
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    main()