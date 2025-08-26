# pdf_main.py
import argparse
import torch

from pdf_processor.llm_loader import LLMHandler
from pdf_processor.utils import setup_logging, validate_steps, load_prompts
from pdf_processor.pipeline import ResearchPaperPipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the MarinePbFusion pdf pipeline.")
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=["filter_pb_papers", "classify_pb_papers", "extract_pb_tables"],
        help="Steps to execute. Valid steps: filter_pb_papers, classify_pb_papers, extract_pb_tables, extract_pb_tables_mineru, extract_pb_tables_query. "
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the LLM model directory"
    )
    parser.add_argument(
        "--load_mode",
        type=str,
        default="transformers",
        help="Model loading mode: 'transformers' or 'vllm'"
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.7,
        help="VLLM GPU memory utilization"
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=16384,
        help="VLLM maximum model length"
    )
    parser.add_argument(
        "--vllm_quantization",
        type=str,
        default="AWQ",
        help="VLLM quantization method"
    )
    parser.add_argument(
        "--prompts_config",
        type=str,
        default="config/prompts.yaml",
        help="YAML file containing prompts configuration"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing PDFs"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="../test/test_pdf.csv",
        help="CSV containing PDF paths"
    )
    parser.add_argument(
        "--pdf_dir",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/test",
        help="Output directory for processed data"
    )

    args = parser.parse_args()

    if not validate_steps(args.steps):
        raise ValueError("Invalid sequence of steps. Please provide a valid sequence as described in the help message.")

    return args

def main():
    """
    Steps:
    - filter_pb_papers
    - classify_pb_papers
    - extract_pb_tables
        - extract_pb_tables_mineru
        - extract_pb_tables_query
    - reorganize_tables_data
    """
    args = parse_arguments()
    setup_logging()
    PROMPTS_CONFIG = load_prompts(args.prompts_config)

    llm_handler = LLMHandler(
        args.model_path,
        load_mode=args.load_mode,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        vllm_quantization=args.vllm_quantization
    )

    pipeline = ResearchPaperPipeline({"prompts": PROMPTS_CONFIG})

    output = None
    for step in args.steps:
        if step == "reorganize_tables_data":
            max_new_tokens = 3200
        elif "extract_pb_tables" in step:
            max_new_tokens = 256
        else:
            max_new_tokens = 128
        generator = llm_handler.create_generator(max_new_tokens = max_new_tokens)
        output = pipeline.process(
            step,
            input_csv = output if output else args.input_csv,
            output_dir=args.output_dir,
            llm_generator=generator,
            batch_size=args.batch_size,
            pdf_dir=args.pdf_dir
        )
    
    if args.load_mode == "vllm":
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()