# pdf_processor/utils.py
import yaml
import logging
import os
from pathlib import Path
import re
import csv


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def validate_steps(steps):
    valid_sequences = [
        ["filter_pb_papers"],
        ["classify_pb_papers"],
        ["extract_pb_tables"],
        ["extract_pb_tables_mineru"],
        ["extract_pb_tables_query"],
        ["reorganize_tables_data"],
        ["filter_pb_papers", "classify_pb_papers"],
        ["filter_pb_papers", "classify_pb_papers", "extract_pb_tables"],
        ["filter_pb_papers", "classify_pb_papers", "extract_pb_tables_mineru"],
        ["filter_pb_papers", "classify_pb_papers", "extract_pb_tables_mineru", "extract_pb_tables_query"],
        ["filter_pb_papers", "classify_pb_papers", "extract_pb_tables", "reorganize_tables_data"],
        ["classify_pb_papers", "extract_pb_tables"],
        ["classify_pb_papers", "extract_pb_tables", "reorganize_tables_data"],
        ["classify_pb_papers", "extract_pb_tables_mineru"],
        ["classify_pb_papers", "extract_pb_tables_mineru", "extract_pb_tables_query"],
        ["extract_pb_tables_mineru", "extract_pb_tables_query"],
        ["extract_pb_tables", "reorganize_tables_data"],
    ]
    if steps in valid_sequences:
        return True
    return False

def load_prompts(config_path: str = None) -> dict:
    """Load prompt templates from YAML file"""
    config_path = Path(__file__).parent.parent / "config/pdf_config/qwen_prompts2.0.yaml" if config_path is None else Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["prompts"]

def truncate_text(text: str, max_length: int) -> str:
    """Intelligently truncate text to specified length by taking the first and last halves at sentence boundaries"""
    if len(text) <= max_length:
        return text

    half_length = max_length // 2
    
    # Find the last sentence boundary in the first half
    first_half_end = text.rfind(".", 0, half_length)
    if first_half_end == -1:
        first_half_end = half_length  # If no boundary, cut at half length
    
    # Find the first sentence boundary in the last half
    last_half_start = text.find(".", len(text) - half_length)
    if last_half_start == -1:
        last_half_start = len(text) - half_length  # If no boundary, cut at half length
    else:
        last_half_start += 1  # Include the period
    
    # Combine the first and last parts
    truncated_text = text[:first_half_end + 1] + " ... " + text[last_half_start:]
    return truncated_text


def save_reorganize_response_to_csv(result_path: str, output_dir: str):
    """Extract CSV content from model output and save to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    success = 0
    success_none = 0
    failed_incomplete = 0
    failed_format = 0
    empty = 0
    
    with open(result_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total += 1
            text = row.get('reorganize_tables_data_response', '')
            pdf_path = row.get('pdf_path', '')
            
            if not text or not pdf_path:
                failed_format += 1
                continue
            
            # Check if we have complete tags
            has_opening = "<answer>" in text
            has_closing = "</answer>" in text

            # Special case: Has closing tag but no opening tag with "Longitude [degrees_east]" present
            if not has_opening and has_closing and "Longitude [degrees_east]" in text:
                # Add opening tag before the first occurrence of "Longitude [degrees_east]"
                text = text.replace("Longitude [degrees_east]", "<answer>Longitude [degrees_east]", 1)
                has_opening = True

            if not has_opening and has_closing and "None" in text:
                text = text.replace("None", "<answer>None", 1)
                has_opening = True

            if has_opening and not has_closing:
                failed_incomplete += 1
                continue

            # Extract content for cases with both tags
            match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
            if not match:
                table_tag_match = re.search(r'<table>(.*?)</table>', text, re.DOTALL)
                if table_tag_match:
                    match = table_tag_match
                elif text != "None" and len(text) > 4 and "none" in text.lower():
                    success_none += 1
                    continue
                elif text == "None":
                    empty += 1
                    continue
                else:
                    # Format issues
                    failed_format += 1
                    pdf_name = os.path.basename(pdf_path)
                    logging.warning(f"{pdf_name}: Cannot extract content due to format issues")
                    continue

            # Extract the content
            csv_content = match.group(1).strip()
            
            # "None" data
            if "none" in csv_content.lower() or not csv_content:
                success_none += 1
                continue
            
            # Success - Process and save
            try:
                rows = csv_content.strip().split('\n')
                if not rows:
                    failed_format += 1
                    continue
                
                csv_filename = os.path.basename(pdf_path).split(".pdf")[0] + ".csv"
                output_path = os.path.join(output_dir, csv_filename)

                header = rows[0].split(',')
                expected_cols = len(header)
                with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    for row_text in rows:
                        # Clean and normalize cells
                        cells = [c.strip().replace('"', '').replace('+', '') 
                                for c in row_text.split(',')]
                        
                        # Ensure correct column count
                        if len(cells) < expected_cols:
                            cells.extend([''] * (expected_cols - len(cells)))
                        elif len(cells) > expected_cols:
                            cells = cells[:expected_cols]
                            
                        writer.writerow(cells)
                
                success += 1
                logging.info(f"Processed {pdf_path} → {output_path}")
                
            except Exception as e:
                failed_format += 1
                logging.error(f"Failed to process {pdf_path}: {str(e)}")

    print(f"Total: {total}\nSuccess Data: {success}, Success None: {success_none}, Failed Incomplete: {failed_incomplete}, Failed Format: {failed_format}, Empty: {empty}")
    return