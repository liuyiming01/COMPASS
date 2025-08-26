# main.py
import os
import json
import logging
import argparse
import pandas as pd
from llm_utils import initialize_model, query_llm, get_conversion_formula, extract_convert_function
from header_mapper import HeaderMapper
from data_standardization import process_data_file, find_data_files
from config import STANDARD_PARAMETERS

def main(model_path, dataset_dir, output_dir):
    # Initialize models
    model, tokenizer = initialize_model(model_path)
    
    # Generate header mappings
    mapper = HeaderMapper(model, tokenizer, output_dir)
    header_mappings = mapper.generate_mappings(dataset_dir)
    with open(f"{output_dir}/header_mappings.json", 'w') as f:
        json.dump(header_mappings, f)

    standard_columns = ['Source_File'] + STANDARD_PARAMETERS
    conversion_cache = {}
    processed_dfs = []

    for data_file in find_data_files(dataset_dir):
        df = process_data_file(
            data_file, column_mappings, standard_columns,
            model, tokenizer, conversion_cache
        )
        if not df.empty:
            processed_dfs.append(df)
        else:
            logging.warning(f"Skipped {os.path.basename(data_file)} due to errors")

    if not processed_dfs:
        logging.error("No data processed. Exiting.")
        return

    combined_df = pd.concat(processed_dfs, ignore_index=True).drop_duplicates()
    combined_df.to_csv(f"{output_dir}/results.csv", index=False)
    logging.info(f"Data saved to {output_dir}/results.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process oceanographic data.')
    parser.add_argument('--model_path', default='Qwen/Qwen2.5-32B-Instruct', help='Path to LLM model')
    parser.add_argument('--dataset_dir', help='Directory containing data CSV files')
    parser.add_argument('--output_dir', help='Path to output merged CSV file')

    args = parser.parse_args()
    
    main(args.model_path, args.dataset_dir, args.output_dir)