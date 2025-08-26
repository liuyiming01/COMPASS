import os

import torch
import pandas as pd
import json
import csv
import re
from typing import List, Tuple
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append('COMPASS/Compass/src/utils')
from dataloader import Dataloader
from prompt import generate_prompt
from metric import evaluate_tuple_extraction, save_tuple_evaluation_results
from table_utils import json_to_markdown, save_response_to_csv


MAX_INPUT_LENGTH = 1200  # Set maximum input length
MAX_ABSTRACT_LENGTH = 200  # Set the maximum length of the abstract

def truncate_text(text: str, max_length: int, tokenizer) -> str:
    """Truncate the text to fit the maximum input length of the model."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_length:
        return text

    truncated_tokens = tokens[:max_length]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    # Ensure that the input does not exceed the model limit.

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                      max_length=MAX_INPUT_LENGTH).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = output_ids[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()


def main():
    TASK = "E2EDataExtraction"
    timestamp = datetime.now().strftime("%m%d_%H%M")

    model_name = "daven3/k2"
    test_data_path = "COMPASS/Compass/evaluation/test_data/test_data.csv"
    mineru_dir = "COMPASS/Compass/evaluation/test_data/mineru"
    true_csv_dir = "COMPASS/Compass/evaluation/test_data/e2e_data"

    output_dir = f"./results/e2e_result_{timestamp}"
    result_csv_path = f"{output_dir}/e2e_generation.csv"
    pred_csv_dir = os.path.join(output_dir, f"e2e_data")
    os.makedirs(pred_csv_dir, exist_ok=True)
    setting_file = os.path.join(output_dir, "setting.json")

    setting = {
        "task": TASK,
        "model_name": model_name,
        "test_data": test_data_path,
        "mineru_dir": mineru_dir,
        "true_e2e_data_dir": true_csv_dir,
    }
    with open(setting_file, 'w', encoding='utf-8') as f:
        json.dump(setting, f, indent=4, ensure_ascii=False)

    print("=" * 70)
    print("End-to-End Extraction Evaluation")
    print("=" * 70)

    print("Loading test data...")
    test_data = Dataloader(test_data_path)
    print(f"Loaded {len(test_data)} test samples")

    print(f"Loading {os.path.basename(model_name)} model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    # Set special tokens
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    print(f"Model loaded successfully. Device: {model.device}")

    fieldnames = ['pdf_path', 'E2EDataExtraction_response']
    with open(result_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
            title = row['title']
            abstract = row['abstract']
            pdf_path = row['pdf_path']
            pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
            raw_label = row['raw_label']

            json_path = os.path.join(mineru_dir, pdf_name, "tables",
                                    f"{pdf_name}_content_list_with_table.json")

            table_list = json_to_markdown(json_path)
            tables = "\n".join(table_list) if table_list else "No tables found."

            abstract = truncate_text(abstract, MAX_ABSTRACT_LENGTH, tokenizer)
            prompt = generate_prompt(TASK, title=title, abstract=abstract, tables=tables)
            response = generate_response(model, tokenizer, prompt, max_new_tokens=2048)

            writer.writerow({
                'pdf_path': pdf_path,
                'E2EDataExtraction_response': response
            })

    save_response_to_csv(result_csv_path, pred_csv_dir)

    tuple_results = evaluate_tuple_extraction(true_csv_dir, pred_csv_dir, test_data)
    
    summary_file, details_file = save_tuple_evaluation_results(tuple_results, output_dir)
    
    agg = tuple_results['aggregate_metrics']
    overall = tuple_results['overall_metrics']
    
    print(f"\nTotal Data Points (Tuples):")
    print(f"  True: {agg['true_tuples']}, Predicted: {agg['pred_tuples']}")
    print(f"  True Positives (TP): {agg['tp']}")
    print(f"  False Positives (FP): {agg['fp']}")
    print(f"  False Negatives (FN): {agg['fn']}")
    print(f"\nOverall Metrics:")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1 Score:  {overall['f1']:.4f}")
    
    print(f"\nResults saved to:")
    print(f"Summary: {summary_file}")
    print(f"Details: {details_file}")

if __name__ == "__main__":
    main()