import os

import torch
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI

import sys
sys.path.append('COMPASS/Compass/src/utils')
from dataloader import Dataloader
from prompt import generate_prompt, extract_prediction_from_response
from metric import calculate_metrics, save_metrics_to_json


def generate_response(prompt: str, max_new_tokens: int = 32) -> str:
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
    TASK = "PaperClassification"
    timestamp = datetime.now().strftime("%m%d_%H%M")

    test_data_path = "COMPASS/Compass/evaluation/test_data/test_data.csv"

    output_dir = f"./results/paper_result_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    predictions_csv_file = f"{output_dir}/paper_predictions.csv"
    metrics_file = f"{output_dir}/paper_metrics.json"
    setting_file = os.path.join(output_dir, "setting.json")

    setting = {
        "task": TASK,
        "model_name": "GPT-4o",
        "test_data": test_data_path,
    }
    with open(setting_file, 'w', encoding='utf-8') as f:
        json.dump(setting, f, indent=4, ensure_ascii=False)


    print("=" * 70)
    print("Paper Classification Evaluation")
    print("=" * 70)

    print("Loading test data...")
    test_data = Dataloader(test_data_path)
    print(f"Loaded {len(test_data)} test samples")
    true_labels = test_data['label'].tolist()

    results = []
    predicted_labels = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        title = row['title']
        abstract = row['abstract']
        true_label = row['label']

        # Generate prompt and get response
        prompt = generate_prompt(task=TASK, title=title, abstract=abstract)
        response = generate_response(prompt, max_new_tokens=32)

        # Extract prediction
        prediction = extract_prediction_from_response(response=response)
        predicted_labels.append(prediction)

        result = {
            'pdf_path': row['pdf_path'],
            'raw_response': response,
            'predicted_label': prediction,
            'true_label': true_label,
        }
        results.append(result)

        if (idx + 1) % 10 == 0:
            correct = sum(1 for i in range(idx + 1) 
                        if predicted_labels[i] is not None and predicted_labels[i] == true_labels[i])
            current_accuracy = correct / (idx + 1)
            
            print(f"Progress: {idx + 1}/{len(test_data)}, Current accuracy: {current_accuracy:.4f}")

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