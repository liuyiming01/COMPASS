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
from openai import OpenAI

import sys
sys.path.append('COMPASS/Compass/src/utils')
from dataloader import Dataloader
from prompt import generate_prompt
from metric import evaluate_tuple_extraction, save_tuple_evaluation_results
from table_utils import json_to_markdown, save_response_to_csv

def generate_response(prompt: str, max_new_tokens: int = 2048) -> str:
    client = OpenAI(
        base_url="API-URL",
        api_key="Your-API-key"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_new_tokens,
        temperature=0
    )
    return response.choices[0].message.content

def main():
    TASK = "E2EDataExtraction"
    timestamp = datetime.now().strftime("%m%d_%H%M")

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
        "model_name": "GPT-4o",
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

            prompt = generate_prompt(TASK, title=title, abstract=abstract, tables=tables)
            response = generate_response(prompt, max_new_tokens=2048)

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