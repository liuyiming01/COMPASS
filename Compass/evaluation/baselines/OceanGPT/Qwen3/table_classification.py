import os

import torch
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append('COMPASS/Compass/src/utils')
from dataloader import Dataloader
from prompt import generate_prompt, extract_prediction_from_response
from metric import calculate_metrics, save_metrics_to_json
from table_utils import format_table_content


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
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
    TASK = "TableClassification"
    timestamp = datetime.now().strftime("%m%d_%H%M")

    model_name = "OceanGPT/OceanGPT-basic-8B"
    test_data_path = "COMPASS/Compass/evaluation/test_data/test_data.csv"
    mineru_dir = "COMPASS/Compass/evaluation/test_data/mineru"

    output_dir = f"./results/table_result_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    predictions_csv_file = f"{output_dir}/table_predictions.csv"
    metrics_file = f"{output_dir}/table_metrics.json"
    setting_file = f"{output_dir}/setting.json"

    setting = {
        "task": TASK,
        "model_name": model_name,
        "test_data": test_data_path,
        "mineru_dir": mineru_dir,
    }
    with open(setting_file, 'w', encoding='utf-8') as f:
        json.dump(setting, f, indent=4, ensure_ascii=False)

    print("=" * 70)
    print("Table Classification Evaluation")
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

    results = []
    true_labels = []
    predicted_labels = []
    sample_count = 0
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        title = row['title']
        abstract = row['abstract']
        pdf_path = row['pdf_path']
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')

        json_dir = os.path.join(
            mineru_dir,
            pdf_name,
            "tables",
        )
        labeled_json_path = os.path.join(json_dir, f"{pdf_name}_table_label.json")
        if os.path.isfile(labeled_json_path):
            json_path = labeled_json_path
        else:
            json_path = os.path.join(json_dir, f"{pdf_name}_content_list_with_table.json")

        with open(json_path, "r", encoding="utf-8") as f:
            json_content = json.load(f)
        table_data = [item for item in json_content if item.get("type") == "table"]

        for i, table in enumerate(table_data):
            table_caption = table.get('table_caption', '')
            table_body = table.get('table_body', '')
            table_footnote = table.get('table_footnote', '')
            true_label = table.get('label', 'Other')
            
            table_content = format_table_content(table_caption, table_body, table_footnote)

            # Generate prompt and get response
            prompt = generate_prompt(task=TASK, title=title,abstract=abstract,table_content=table_content)
            response = generate_response(model, tokenizer, prompt, max_new_tokens=32)

            # Extract prediction
            prediction = extract_prediction_from_response(response)
            
            true_labels.append(true_label)
            predicted_labels.append(prediction)

            result = {
                'pdf_path': pdf_path,
                'table_index': i,
                'raw_response': response,
                'predicted_label': prediction,
                'true_label': true_label,
            }
            results.append(result)
            sample_count += 1

            if sample_count % 10 == 0:
                correct = sum(1 for j in range(sample_count) 
                            if predicted_labels[j] is not None and predicted_labels[j] == true_labels[j])
                current_accuracy = correct / sample_count

                print(f"Progress: {sample_count} tables processed, Current accuracy: {current_accuracy:.4f}")

    print(f"\nTotal tables processed: {sample_count}")
    print(f"Label distribution: {pd.Series(true_labels).value_counts().to_dict()}")

    # Save predictions
    df_results = pd.DataFrame(results)
    df_results.to_csv(predictions_csv_file, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_csv_file}")

    print("EVALUATION RESULTS")
    metrics = calculate_metrics(TASK, true_labels, predicted_labels)
    save_metrics_to_json(metrics, metrics_file)

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Macro F1 (excl. 'Other'): {metrics.get('macro_f1_excl_other', 'N/A'):.4f}")
    print(f"\nParsing Success Rate: {metrics['parsing_success_rate']:.4f}")

    return results, metrics


if __name__ == "__main__":
    results, metrics = main()