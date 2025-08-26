# table_processor/unit_converter.py
import logging
import os
import re
import numpy as np
from typing import Callable, Dict, Optional, List
import pandas as pd
import inspect

from .utils import get_parameters, get_prompts, extract_unit_from_column

logger = logging.getLogger(__name__)

class UnitConverter:
    """Handle unit conversions for standardized parameters."""
    def __init__(self, llm_generator: Optional[Callable] = None):
        self.llm_generator = llm_generator
        self.params = get_parameters()
        self.prompts = get_prompts()

    def convert_units(self, csv_path: str, output_dir: str, mapping: Dict) -> pd.DataFrame:
        """Convert units in the CSV file based on the provided mapping."""
        if not self.llm_generator:
            raise RuntimeError("Must provide llm_generator to perform unit conversion")
        df = pd.read_csv(csv_path)
        # Remove columns that are completely empty.
        df.dropna(axis=1, how='all', inplace=True)
        df = df.applymap(self._clean_value)

        records = []
        prompt_template = self.prompts["unit_conversion"]
        conversion_cache = {}  # key: (orig_unit_lower, std_unit_lower) -> (convert_func, response)

        # Load existing conversion records for this CSV file if available
        unit_log_file = os.path.join(output_dir, "unit_conversion_records.csv")
        current_file = os.path.basename(csv_path)
        if os.path.exists(unit_log_file):
            try:
                stored = pd.read_csv(unit_log_file)
                subset = stored[stored['data_file'] == current_file]
                for rec in subset.to_dict(orient='records'):
                    orig_col_stored = rec["original_column"]
                    std_col_stored = rec["standard_column"]
                    orig_unit = extract_unit_from_column(orig_col_stored)
                    std_unit = extract_unit_from_column(std_col_stored)
                    key = (orig_unit.lower(), std_unit.lower())
                    if key not in conversion_cache:
                        conversion_cache[key] = (self._get_conversion_func(rec["model_response"]), rec["model_response"])
            except Exception as e:
                logger.error("Error reading unit conversion records: %s", str(e))

        for orig_col, std_col in mapping.items():
            if orig_col not in df.columns:
                continue

            orig_unit = extract_unit_from_column(orig_col)
            std_unit = extract_unit_from_column(std_col)
            if orig_unit and std_unit and not self._units_match(orig_unit, std_unit):
                cache_key = (orig_unit.lower(), std_unit.lower())
                if cache_key not in conversion_cache:
                    prompt = prompt_template.format(
                        original_unit=orig_unit,
                        standard_unit=std_unit,
                    )
                    response = self.llm_generator([prompt])[0]
                    convert_func = self._get_conversion_func(response)
                    conversion_cache[cache_key] = (convert_func, response)

                    records.append({
                        "data_file": current_file,
                        "original_column": orig_col,
                        "standard_column": std_col,
                        "model_response": response
                    })
                else:
                    convert_func, response = conversion_cache[cache_key]
                
                if convert_func:
                    try:
                        params = inspect.signature(convert_func).parameters
                        if len(params) == 1:
                            df[orig_col] = df[orig_col].apply(convert_func)
                            logger.info(f"Converted {orig_col} from {orig_unit} to {std_unit}")
                        else:
                            logger.error(f"Invalid conversion function for {orig_unit}→{std_unit}: expected 1 parameter, got {len(params)}")
                    except Exception as e:
                        logger.error(f"Error inspecting conversion function for {orig_unit}: {e}")
                else:
                    logger.warning(f"No conversion function found for {orig_unit} to {std_unit}")

            if orig_col != std_col:
                df.rename(columns={orig_col: std_col}, inplace=True)

        if records:
            self._save_records(unit_log_file, records)

        # Ensure DataFrame contains all standard columns, adding missing ones with empty values
        for col in self.params.get("standard_columns", []):
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def _save_records(self, log_file: str, records: List[dict]) -> None:
        """Save records to CSV file."""
        header = not os.path.exists(log_file)
        pd.DataFrame(records).to_csv(log_file, mode='a', header=header, index=False)

    def _units_match(self, unit1: str, unit2: str) -> bool:
        return unit1 and unit2 and unit1.lower() == unit2.lower()

    def _get_conversion_func(self, response: str):
        pattern = r"<func>(.*?)</func>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            func_code = match.group(1).strip()
            # Check if the extracted code is enclosed within triple backticks
            if func_code.startswith("```"):
                # Remove the opening ```python or ``` if present
                func_code = re.sub(r'^```(?:python)?\s*', '', func_code, flags=re.IGNORECASE).strip()
                # Remove the closing ```
                func_code = re.sub(r'\s*```$', '', func_code).strip()
            local_namespace = {}

            try:
                exec(func_code, {}, local_namespace)
                # 自动选取代码中定义的第一个可调用对象
                funcs = [v for v in local_namespace.values() if callable(v)]
                if funcs:
                    return funcs[0]
                else:
                    logger.warning("No callable found in conversion function code.")
            except Exception as e:
                logger.error("Error executing conversion function code: %s", e)
                return None
        else:
            logger.warning("No <func> tag found in the model output.")
        return None
    
    def _clean_value(self, value):
        """
        Clean a cell value so that it becomes a decimal number or NaN.
        For non–pure-numeric cells, handle two cases:
        1. If there is a plus–minus pattern (±) between numbers, remove the ± and everything after it.
        2. If there are parentheses (round or square), remove the parentheses and their content.
        """
        if pd.isna(value):
            return np.nan
        s = str(value).strip()
        s = s.replace(',', '')
        # Check if it's already a valid decimal
        if re.fullmatch(r'[+-]?(\d+(\.\d+)?|\.\d+)', s):
            try:
                return float(s)
            except ValueError:
                return np.nan
        # Case 1: Remove plus–minus patterns (e.g., "123.45 ± 0.67" -> "123.45")
        s = re.sub(r'([+-]?\d+(?:\.\d+)?)\s*[±]\s*[+-]?\d+(?:\.\d+)?', r'\1', s)
        # Case 2: Remove any parentheses (round or square) and their contents
        s = re.sub(r'\s*[\(\[].*?[\)\]]', '', s)
        s = s.strip()
        # Try to convert cleaned string to float
        if re.fullmatch(r'[+-]?(\d+(\.\d+)?|\.\d+)', s):
            try:
                return float(s)
            except ValueError:
                return np.nan
        return np.nan