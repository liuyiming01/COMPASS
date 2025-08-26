import os

import torch
import pandas as pd
import json
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append('COMPASS/Compass/src/utils')
from dataloader import Dataloader
from prompt import generate_ablation_prompt, extract_prediction_from_response
from metric import calculate_metrics, save_metrics_to_json


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


def kTree_paper_classification(model, tokenizer, title: str, abstract: str) -> Tuple[str, str, str, str]:
    """
    Perform kTree paper classification: Pb_Related_Paper first, then PaperClassification
    """
    pb_related_prompt, classification_prompt = generate_ablation_prompt(task='wo_knowledge_node_PaperClassification',title=title,abstract=abstract)
    pb_related_response = generate_response(model, tokenizer, pb_related_prompt)
    pb_related_prediction = extract_prediction_from_response(pb_related_response)

    if pb_related_prediction == "YES":
        classification_response = generate_response(model, tokenizer, classification_prompt)
        final_prediction = extract_prediction_from_response(classification_response)
        final_response = classification_response
    else:
        final_prediction = "Other" if pb_related_prediction == "NO" else None
        final_response = ""
    
    return pb_related_prediction, pb_related_response, final_prediction, final_response


def main():
    TASK = "wo_knowledge_node_PaperClassification"
    timestamp = datetime.now().strftime("%m%d_%H%M")

    model_name = "Qwen/Qwen3-8B"
    test_data_path = "COMPASS/Compass/evaluation/test_data/test_data.csv"

    output_dir = f"./results/paper_result_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    predictions_csv_file = f"{output_dir}/paper_predictions.csv"
    metrics_file = f"{output_dir}/paper_metrics.json"
    setting_file = f"{output_dir}/setting.json"

    setting = {
        "task": TASK,
        "model_name": model_name,
        "test_data": test_data_path,
    }
    with open(setting_file, 'w', encoding='utf-8') as f:
        json.dump(setting, f, indent=4, ensure_ascii=False)
    
    print("=" * 70)
    print("Knowledge Tree Paper Classification Evaluation")
    print("=" * 70)

    print("Loading test data...")
    test_data = Dataloader(test_data_path)
    print(f"Loaded {len(test_data)} test samples")
    true_labels = test_data['label'].tolist()

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
    pb_related_predictions = []
    final_predictions = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        title = row['title']
        abstract = row['abstract']
        true_label = row['label']

        pb_related_pred, pb_related_resp, final_pred, final_resp = kTree_paper_classification(
            model, tokenizer, title, abstract
        )

        pb_related_predictions.append(pb_related_pred)
        final_predictions.append(final_pred)

        result = {
            'pdf_path': row['pdf_path'],
            'true_label': true_label,
            'pb_related_response': pb_related_resp,
            'pb_related_prediction': pb_related_pred,
            'final_response': final_resp,
            'final_prediction': final_pred,
        }
        results.append(result)

        if (idx + 1) % 10 == 0:
            correct = sum(1 for i in range(idx + 1) 
                        if final_predictions[i] is not None and final_predictions[i] == true_labels[i])
            current_accuracy = correct / (idx + 1)
            print(f"Progress: {idx + 1}/{len(test_data)}, Current accuracy: {current_accuracy:.4f}")

    # Save predictions
    df_results = pd.DataFrame(results)
    df_results.to_csv(predictions_csv_file, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_csv_file}")

    print("EVALUATION RESULTS")
    # Step 1: Pb_Related metrics
    print("\nStep 1: Pb_Related_Paper Classification Results")
    print("-" * 50)
    true_pb_related = ["YES" if label in ["Marine_Pb", "Atmospheric_Pb", "Terrestrial_Pb"] else "NO" 
                       for label in true_labels]
    pb_related_metrics = calculate_metrics('Pb_Related_Paper', true_pb_related, pb_related_predictions)

    # Step 2: Final classification metrics
    print("\nStep 2: Final Paper Classification Results")
    print("-" * 50)
    final_metrics = calculate_metrics('PaperClassification', true_labels, final_predictions)
    
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"  Macro F1 (excl. 'Other'): {final_metrics.get('macro_f1_excl_other', 'N/A'):.4f}")
    print(f"\nParsing Success Rate: {final_metrics['parsing_success_rate']:.4f}")

    combined_metrics = {
        'Pb_Related_Paper_metrics': pb_related_metrics,
        'PaperClassification_metrics': final_metrics,
    }
    save_metrics_to_json(combined_metrics, metrics_file)

    return results, combined_metrics

if __name__ == "__main__":
    results, metrics = main()