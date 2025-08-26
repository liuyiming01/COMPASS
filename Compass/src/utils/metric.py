import os
import json
import yaml
import re
import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
from collections import defaultdict


LABEL_MAPPINGS = {
        'PaperClassification': ["Marine_Pb", "Atmospheric_Pb", "Terrestrial_Pb", "Other"],
        'TableClassification': ["Pb_CONC", "210Pb_CONC", "Pb_RATIO", "Other"],
        'Pb_Related_Paper': ["YES", "NO"]
    }

def calculate_metrics(task: str, true_labels: List[str], predicted_labels: List[str]) -> Dict:
    """
    Calculate all metrics with configurable handling of parsing failures
    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels (may contain None for parsing failures)
        task: 'PaperClassification', 'TableClassification', or 'Pb_Related'
    """
    total_samples = len(predicted_labels)
    parsing_failures = sum(1 for pred in predicted_labels if pred is None)
    parsing_success_rate = (total_samples - parsing_failures) / total_samples if total_samples > 0 else 0.0
    
    main_labels = LABEL_MAPPINGS[task]
    valid_label_set = set(main_labels)
    pred_processed = [
        pred if (pred is not None and pred in valid_label_set) else 'PARSE_FAILED'
        for pred in predicted_labels
    ]
    accuracy = accuracy_score(true_labels, pred_processed)
    macro_f1 = f1_score(true_labels, pred_processed, labels=main_labels, average='macro', zero_division=0)
    per_class_f1 = f1_score(true_labels, pred_processed, labels=main_labels, average=None, zero_division=0)
    per_class_f1_dict = {label: f1 for label, f1 in zip(main_labels, per_class_f1)}
    
    macro_f1_excl_other = None
    if task in ['PaperClassification', 'TableClassification']:
        labels_excl_other = [label for label in main_labels if label != 'Other']
        macro_f1_excl_other = f1_score(true_labels, pred_processed, labels=labels_excl_other, average='macro', zero_division=0)

    return {
        'task': task,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_f1_excl_other': macro_f1_excl_other,
        'parsing_success_rate': parsing_success_rate,
        'per_class_f1': per_class_f1_dict,
        'labels_order': main_labels,
        'total_samples': total_samples,
    }

def save_metrics_to_json(metrics: Dict, output_path: str):
    """Save metrics to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {output_path}")


# E2EDataExtraction Metric

REQUIRED_COLUMNS = [
    'Longitude [degrees_east]',
    'Latitude [degrees_north]',
    'DEPTH [m]'
]
HEADER_MAP = {
    'longitude': 'Longitude [degrees_east]',
    'longitude [degrees_east]': 'Longitude [degrees_east]',
    'lon': 'Longitude [degrees_east]',
    'latitude': 'Latitude [degrees_north]',
    'latitude [degrees_north]': 'Latitude [degrees_north]',
    'lat': 'Latitude [degrees_north]',
    'depth': 'DEPTH [m]',
    'depth [m]': 'DEPTH [m]',
}

def standardize_headers(df):
    new_columns = []
    for col in df.columns:
        key = col.lower().strip()
        new_columns.append(HEADER_MAP.get(key, col))
    df.columns = new_columns
    return df

def extract_leading_number(cell):
    if pd.isna(cell):
        return np.nan
    try:
        # 支持科学计数法和负号
        return float(re.search(r"^\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", str(cell)).group(1))
    except (AttributeError, ValueError):
        return np.nan

def extract_tuples_from_csv(csv_path: str) -> list:
    tuples = []
    
    df = pd.read_csv(csv_path)
    df = standardize_headers(df)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns in {csv_path}: {missing_columns}")
        return []

    pb_columns = [col for col in df.columns 
                    if col not in REQUIRED_COLUMNS and 'pb' in col.lower()]
    
    if not pb_columns:
        print(f"No Pb columns found in {csv_path}.")
        return []

    for _, row in df.iterrows():
        lon = extract_leading_number(row.get(REQUIRED_COLUMNS[0], np.nan))
        lat = extract_leading_number(row.get(REQUIRED_COLUMNS[1], np.nan))
        depth = extract_leading_number(row.get(REQUIRED_COLUMNS[2], np.nan))
        
        if pd.isna(lon) or pd.isna(lat) or pd.isna(depth):
            continue

        for pb_col in pb_columns:
            pb_value = extract_leading_number(row.get(pb_col, np.nan))
            if not pd.isna(pb_value):
                tuples.append((
                    f"{float(lon):.3f}",
                    f"{float(lat):.3f}",
                    f"{float(depth):.1f}",
                    f"{float(pb_value):.3f}"
                ))
    return tuples

def normalize_tuple(t: tuple) -> tuple:
    lon, lat, depth, pb = t

    try:
        lon_val = float(lon)
        if lon_val > 180:
            lon_val -= 360
        lon_norm = f"{lon_val:.3f}"
    except Exception:
        lon_norm = str(lon).strip()

    def norm_num(val, fmt):
        try:
            return fmt.format(float(val))
        except Exception:
            return str(val).strip()

    lat_norm = norm_num(lat, "{:.3f}")
    depth_norm = norm_num(depth, "{:.1f}")
    pb_norm = norm_num(pb, "{:.3f}")

    return (lon_norm, lat_norm, depth_norm, pb_norm)

def calculate_tuple_metrics(true_tuples: list, pred_tuples: list) -> dict:
    true_set = set(normalize_tuple(t) for t in true_tuples)
    pred_set = set(normalize_tuple(t) for t in pred_tuples)
    
    tp = true_set & pred_set
    fp = pred_set - true_set
    fn = true_set - pred_set
    
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'true_tuples': len(true_tuples),
        'pred_tuples': len(pred_tuples),
        'tp': len(tp),
        'fp': len(fp),
        'fn': len(fn),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_tuple_extraction(true_csv_dir: str, pred_csv_dir: str, test_data: pd.DataFrame) -> dict:
    results = {
        'total_samples': 0,
        'has_table_samples': 0,
        'no_table_samples': 0,
        'aggregate_metrics': {
            'true_tuples': 0,
            'pred_tuples': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0
        },
        'per_sample_metrics': []
    }

    for idx, row in test_data.iterrows():
        pdf_path = row['pdf_path']
        pdf_name = os.path.basename(pdf_path).replace('.pdf', '')
        results['total_samples'] += 1
        
        true_csv_path = os.path.join(true_csv_dir, f"{pdf_name}.csv")
        has_table = os.path.exists(true_csv_path)
        
        sample_metrics = {
            'pdf_name': pdf_name,
            'has_table': has_table,
            'true_tuples': 0,
            'pred_tuples': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
        }
        
        if has_table:
            results['has_table_samples'] += 1
            
            true_tuples = extract_tuples_from_csv(true_csv_path)
            sample_metrics['true_tuples'] = len(true_tuples)

            pred_csv_path = os.path.join(pred_csv_dir, f"{pdf_name}.csv")
            if os.path.exists(pred_csv_path):
                pred_tuples = extract_tuples_from_csv(pred_csv_path)
                sample_metrics['pred_tuples'] = len(pred_tuples)

                tuple_metrics = calculate_tuple_metrics(true_tuples, pred_tuples)
                sample_metrics.update(tuple_metrics)
            else:
                sample_metrics['fn'] = len(true_tuples)

            results['aggregate_metrics']['true_tuples'] += sample_metrics['true_tuples']
            results['aggregate_metrics']['pred_tuples'] += sample_metrics['pred_tuples']
            results['aggregate_metrics']['tp'] += sample_metrics['tp']
            results['aggregate_metrics']['fp'] += sample_metrics['fp']
            results['aggregate_metrics']['fn'] += sample_metrics['fn']
        
        else:
            results['no_table_samples'] += 1
            
            pred_csv_path = os.path.join(pred_csv_dir, f"{pdf_name}.csv")
            if os.path.exists(pred_csv_path):
                pred_tuples = extract_tuples_from_csv(pred_csv_path)
                sample_metrics['pred_tuples'] = len(pred_tuples)
                sample_metrics['fp'] = len(pred_tuples)
                
                results['aggregate_metrics']['pred_tuples'] += len(pred_tuples)
                results['aggregate_metrics']['fp'] += len(pred_tuples)
        
        results['per_sample_metrics'].append(sample_metrics)

    agg = results['aggregate_metrics']
    precision = agg['tp'] / (agg['tp'] + agg['fp']) if (agg['tp'] + agg['fp']) > 0 else 0
    recall = agg['tp'] / (agg['tp'] + agg['fn']) if (agg['tp'] + agg['fn']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results['overall_metrics'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'data_points_ratio': agg['tp'] / agg['true_tuples'] if agg['true_tuples'] > 0 else 0
    }
    
    return results

def save_tuple_evaluation_results(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'total_samples': results['total_samples'],
        'has_table_samples': results['has_table_samples'],
        'no_table_samples': results['no_table_samples'],
        'aggregate_metrics': results['aggregate_metrics'],
        'overall_metrics': results['overall_metrics']
    }
    
    summary_file = os.path.join(output_dir, f"tuple_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)

    details_file = os.path.join(output_dir, f"tuple_details.csv")
    pd.DataFrame(results['per_sample_metrics']).to_csv(details_file, index=False)
    
    return summary_file, details_file

