import os
import re
import pandas as pd
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

def process_data_file(
    data_file: str,
    column_mappings: Dict,
    standard_cols: List[str],
    model,
    tokenizer,
    conversion_cache: Dict
) -> pd.DataFrame:
    """Process a single data file: map columns, clean data, convert units."""
    try:
        df = pd.read_csv(data_file)
        data_filename = os.path.basename(data_file)
        params = pd.read_csv(get_parameters_path(data_file))
        param_units = dict(zip(params['Supplied Name'], params['Supplied Units']))
        mapping = column_mappings.get(data_filename, {})
        
        # Rename columns and retain only standard columns
        df.rename(columns=mapping, inplace=True)
        df = df.reindex(columns=standard_cols)
        
        # Clean invalid data
        df = handle_invalid_data(df)
        
        # Unit conversion for each applicable column
        for std_col in df.columns:
            if std_col == 'Source_File':
                continue
            std_unit = parse_standard_unit(std_col)
            orig_col = next((k for k, v in mapping.items() if v == std_col), None)
            orig_unit = param_units.get(orig_col, '') if orig_col else None
            
            if not std_unit or not orig_unit or std_unit == orig_unit:
                continue
            
            cache_key = (orig_unit, std_unit)
            if cache_key not in conversion_cache:
                prompt = get_conversion_formula(orig_unit, std_unit, std_col)
                llm_output = query_llm(prompt, model, tokenizer)
                conversion_cache[cache_key] = extract_convert_function(llm_output)
            
            convert_func = conversion_cache[cache_key]
            if convert_func:
                try:
                    df[std_col] = df[std_col].apply(
                        lambda x: convert_func(x) if pd.notna(x) else x
                    )
                except Exception as e:
                    logging.error(f"Unit conversion failed for {std_col}: {e}")
                    df[std_col] = pd.NA
            else:
                df[std_col] = pd.NA
        
        df['Source_File'] = data_filename
        return df
    except Exception as e:
        logging.error(f"Error processing {data_file}: {e}")
        return pd.DataFrame()

def get_parameters_path(data_file: str) -> str:
    """Get the path to the parameters file for a data file."""
    base = os.path.splitext(data_file)[0]
    param_file = f"{base}_Parameters.csv"
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file missing: {param_file}")
    return param_file

def handle_invalid_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid data markers with NaN and convert to numeric."""
    invalid = ['-999']
    df.replace(invalid, pd.NA, inplace=True)

    for col in df.columns:
        if col != 'Source_File':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def parse_standard_unit(column: str) -> str:
    """Extract unit from standard column name (e.g., '[pmol/kg]')."""
    match = re.search(r'\[(.*?)\]$', column)
    return match.group(1) if match else None

def find_data_files(data_dir: str) -> List[str]:
    """Find all data CSV files excluding parameter files."""
    return [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.csv') and not f.endswith('_Parameters.csv')
    ]