import os

import torch
import pandas as pd
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from datetime import datetime

import sys
sys.path.append('COMPASS/Compass/src/utils')
from dataloader import Dataloader
from prompt import generate_prompt, extract_prediction_from_response
from metric import calculate_metrics, save_metrics_to_json


MAX_INPUT_LENGTH = 2000  # Set maximum input length
MAX_ABSTRACT_LENGTH = 1500  # Set the maximum length of the abstract

def truncate_text(text: str, max_length: int, tokenizer) -> str:
    """Truncate the text to fit the maximum input length of the model."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_length:
        return text

    truncated_tokens = tokens[:max_length]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return truncated_text


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
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
    TASK = "PaperClassification"
    timestamp = datetime.now().strftime("%m%d_%H%M")

    model_name = "daven3/k2"
    test_data_path = "COMPASS/Compass/evaluation/test_data/test_data.csv"

    output_dir = f"./results/paper_result_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    predictions_csv_file = f"{output_dir}/paper_predictions.csv"
    metrics_file = f"{output_dir}/paper_metrics.json"
    setting_file = os.path.join(output_dir, "setting.json")

    setting = {
        "task": TASK,
        "model_name": model_name,
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

    results = []
    predicted_labels = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        title = row['title']
        abstract = row['abstract']
        true_label = row['label']

        # Generate prompt and get response
        abstract = truncate_text(abstract, MAX_ABSTRACT_LENGTH, tokenizer)
        prompt = generate_prompt(task=TASK, title=title, abstract=abstract)
        response = generate_response(model, tokenizer, prompt, max_new_tokens=32)

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