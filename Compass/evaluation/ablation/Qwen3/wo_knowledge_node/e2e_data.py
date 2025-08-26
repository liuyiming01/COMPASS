import os

import torch
import pandas as pd
import json
import csv
import re
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime

from paper_classification import kTree_paper_classification

import sys
sys.path.append('COMPASS/Compass/src/utils')
from dataloader import Dataloader
from prompt import generate_ablation_prompt, extract_prediction_from_response
from metric import calculate_metrics, save_metrics_to_json, evaluate_tuple_extraction, save_tuple_evaluation_results
from table_utils import json_to_markdown, save_response_to_csv, format_table_content


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 2048) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return content


def main():
    TASK = "wo_knowledge_node_E2EDataExtraction"
    timestamp = datetime.now().strftime("%m%d_%H%M")

    model_name = "Qwen/Qwen3-8B"
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
    print("Knowledge Tree End-to-End Extraction Evaluation")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

            # Step 1: Paper Classification
            _, _, paper_pred, _ = kTree_paper_classification(model, tokenizer, title, abstract)
            if paper_pred != "Marine_Pb":
                writer.writerow({
                    'pdf_path': pdf_path,
                    'E2EDataExtraction_response': "None"
                })
                continue

            json_path = os.path.join(mineru_dir, pdf_name, "tables",
                                    f"{pdf_name}_content_list_with_table.json")

            table_list = json_to_markdown(json_path)
            tables_str = "\n".join(table_list) if table_list else "No tables found."
            prompt = generate_ablation_prompt(TASK, tables=tables_str)
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
    print(f"  Data Points Ratio (TP/True): {overall['data_points_ratio']:.2%}")
    
    print(f"\nResults saved to:")
    print(f"Summary: {summary_file}")
    print(f"Details: {details_file}")


if __name__ == "__main__":
    main()