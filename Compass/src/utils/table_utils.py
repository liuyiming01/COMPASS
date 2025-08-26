from typing import List, Tuple
import re
from bs4 import BeautifulSoup
import os
import json
import csv

def format_table_content(table_caption: str, table_body: str, table_footnote: str) -> str:
    """Format table content for prompt input"""
    content_parts = []
    if table_caption:
        caption_text = ''.join(table_caption) if isinstance(table_caption, list) else table_caption
        if caption_text.strip():
            content_parts.append(f"Table Caption: {caption_text}")
    if table_body:
        table_summary = parse_html_table(table_body, max_rows=5)
        if table_summary:
            content_parts.append(f"Table Data:\n{table_summary}")
    if table_footnote:
        footnote_text = ''.join(table_footnote) if isinstance(table_footnote, list) else table_footnote
        if footnote_text.strip():
            content_parts.append(f"Table Notes: {footnote_text}")
    return '\n'.join(content_parts)


def parse_html_table(html_string: str, max_rows: int = 3) -> str:
    """Parse HTML table and extract key information"""
    if not html_string or not html_string.strip():
        return ""
    html_string = html_string.strip('\n')

    try:
        soup = BeautifulSoup(html_string, 'html.parser')
        table = soup.find('table')
        if not table:
            clean_text = re.sub(r'<[^>]+>', ' ', html_string)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            return clean_text[:500] + "..." if len(clean_text) > 500 else clean_text

        # Extract headers
        headers = []
        header_row = table.find('tr')
        if header_row:
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]

        # Extract data rows (limited)
        data_rows = []
        all_rows = table.find_all('tr')[1:]
        for i, row in enumerate(all_rows[:max_rows]):
            cells = row.find_all(['td', 'th'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            if any(cell for cell in row_data):
                data_rows.append(row_data)

        # Construct summary
        summary_parts = []
        if headers:
            summary_parts.append(" | ".join(headers))
        if data_rows:
            for i, row in enumerate(data_rows):
                row_text = " | ".join(row)
                summary_parts.append(row_text)
        if len(all_rows) > max_rows:
            summary_parts.append(f"... [and {len(all_rows) - max_rows} more rows]")
        return "\n".join(summary_parts)

    except Exception as e:
        print(f"Warning: Error parsing HTML table: {e}")
        clean_text = re.sub(r'<[^>]+>', ' ', html_string)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        return clean_text[:500] + "..." if len(clean_text) > 500 else clean_text

def json_to_markdown(json_path: str) -> Tuple[str, List[str]]:
    """Convert JSON content to formatted markdown for LLM input"""
    def safe_json_load(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    tables = []
    for item in safe_json_load(json_path):
        if item["type"] == "table":
            table_str = format_table(item)
            if table_str:
                tables.append(table_str)
    return tables

def format_table(item: dict) -> str:
    """Format table elements with caption and footnotes"""
    caption = "".join(item.get("table_caption", []))
    body = item.get("table_body", "")
    footnote = "".join(item.get("table_footnote", []))
    
    if not any([caption, body, footnote]):
        return ""
    return rf"{caption}\n{body}\n{footnote}".strip()

def save_response_to_csv(result_path: str, output_dir: str):
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
            text = row.get('E2EDataExtraction_response', '')
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
                text += "</answer>"
                failed_incomplete += 1
                # continue

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
                    print(f"{pdf_name}: Cannot extract content due to format issues")
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
                print(f"Processed {pdf_path} → {output_path}")

            except Exception as e:
                failed_format += 1
                print(f"Failed to process {pdf_path}: {str(e)}")

    print(f"Total: {total}\nSuccess Data: {success}, Success None: {success_none}, Failed Incomplete: {failed_incomplete}, Failed Format: {failed_format}, Empty: {empty}")
    return