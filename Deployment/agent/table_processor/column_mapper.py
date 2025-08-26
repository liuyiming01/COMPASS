# table_processor/header_standardizer.py
import logging
import os
import re
from typing import Callable, Dict, Optional, Tuple, List

import pandas as pd
from .utils import get_parameters, get_prompts, extract_unit_from_column, remove_brackets

logger = logging.getLogger(__name__)

class ColumnMapper:
    """Map CSV columns to standard parameters."""
    def __init__(self, strategy: str = 'rule_based', llm_generator: Optional[Callable] = None):
        self.strategy = strategy
        self.llm_generator = llm_generator
        self.params = get_parameters()
        self.prompts = get_prompts()
        self.standard_columns = self.params["standard_columns"]
        self.mandatory_columns = self.params["mandatory_columns"]

    def standardize(self, csv_path: str, output_dir: str) -> Dict[str, str]:
        """Execute standardization process for a single CSV file."""
        if self.strategy == "rule_based":
            return self._rule_based_mapping(csv_path)
        if self.strategy == "llm":
            return self._llm_based_mapping(csv_path, output_dir)
        raise ValueError(f"Invalid strategy: {self.strategy}")
        
    def _handle_standard_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Process mandatory columns and return initial mapping."""
        mapping = {}
        for col in df.columns:
            if col in self.standard_columns:
                mapping[col] = col
        return mapping

    def _rule_based_mapping(self, csv_path: str) -> Dict[str, str]:
        """Generate column mapping using rule-based approach."""
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        # Remove columns that are completely empty.
        df.dropna(axis=1, how='all', inplace=True)

        mapping = self._handle_standard_columns(df)
        columns_to_process = [col for col in df.columns if col not in mapping]
        for orig_col in columns_to_process:
            matched_col = self._column_match(orig_col)
            if matched_col:
                mapping[orig_col] = matched_col
        return mapping

    def _is_pb_related(self, column_name: str) -> bool:
        """Check if column is Pb-related."""
        return re.search(r'\b(?:pb|lead)\b', column_name, re.IGNORECASE) is not None

    def _column_match(self, column_name: str) -> str:
        """Rule-based matching for Pb columns."""
        if self._is_pb_related(column_name):
            numbers = re.findall(r'\d+', column_name)
            unit = extract_unit_from_column(column_name)
            
            # Isotope ratio detection
            ratios = [('206', '204'), ('206', '207'), ('208', '206'),
                    ('207', '204'), ('208', '207'), ('208', '204')]
            for num, den in ratios:
                if num in numbers and den in numbers:
                    return f'Pb_{num}_{den}'
            if unit:
                if '210' in numbers:
                    return 'Pb_210_CONC [mBq/kg]'
                else:
                    return 'Pb_CONC [pmol/kg]'
        else:
            if "lon" in column_name.lower():
                return 'Longitude [degrees_east]'
            elif "lat" in column_name.lower():
                return 'Latitude [degrees_north]'
            elif "depth" in column_name.lower():
                return 'DEPTH [m]'
        
        return None


    def _llm_based_mapping(self, csv_path: str, output_dir: str) -> Dict[str, str]:
        """LLM-based mapping implementation"""
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        # Remove columns that are completely empty.
        df.dropna(axis=1, how='all', inplace=True)

        mapping = self._handle_standard_columns(df)
        mapping_log_file = os.path.join(output_dir, "mapping_records.csv")

        if os.path.exists(mapping_log_file):
            try:
                stored = pd.read_csv(mapping_log_file)
                current_file = os.path.basename(csv_path)
                subset = stored[stored['data_file'] == current_file]
                if not subset.empty:
                    for _, row in subset.iterrows():
                        if pd.notna(row['matched_param']) and row['matched_param'] in self.standard_columns:
                            mapping[row['column']] = row['matched_param']
                    return mapping
            except Exception as e:
                logger.error("Error reading stored mapping records: %s", str(e))

        records = []
        columns_to_process = [col for col in df.columns if col not in mapping]
        prompt_template = self.prompts['column_mapping']['pdf_column_mapping']
        
        for col in columns_to_process:
            col_processed = remove_brackets(col)
            prompt = prompt_template.format(name=col_processed)
            try:
                response = self.llm_generator([prompt])[0]
                match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                matched_param = match.group(1).strip() if match else None
                if matched_param and matched_param in self.standard_columns:
                    mapping[col] = matched_param
                records.append({
                    'data_file': os.path.basename(csv_path),
                    'column': col,
                    'column_mapping_response': response,
                    'matched_param': matched_param
                })
            except Exception as e:
                logger.error("Failed to process column %s: %s", col, str(e))

        if records:
            self._save_records(mapping_log_file, records)
        return mapping

    def _save_records(self, log_file: str, records: List[dict]) -> None:
        """Save records to CSV file."""
        header = not os.path.exists(log_file)
        pd.DataFrame(records).to_csv(log_file, mode='a', header=header, index=False)