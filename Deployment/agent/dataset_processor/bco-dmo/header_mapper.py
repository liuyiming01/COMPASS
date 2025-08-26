# header_mapper.py
import pandas as pd
import os
import json
import logging
from tqdm import tqdm
from .config import STANDARD_PARAMETERS, PARAMETER_PROMPT_TEMPLATE

class HeaderMapper:
    def __init__(self, model, tokenizer, output_dir, checkpoint_path="mapping_checkpoint.json"):
        self.model = model
        self.tokenizer = tokenizer
        self.standard_params = '\n'.join(f'- {p}' for p in STANDARD_PARAMETERS)
        self.checkpoint_path = f"{output_dir}/{checkpoint_path}"
        self.mappings = self._load_checkpoint()
        self.logger = logging.getLogger(__name__)

        self.geo_params = {param for param in STANDARD_PARAMETERS if any(x in param for x in ['Longitude', 'Latitude', 'DEPTH'])}
        self.pb_params = {param for param in STANDARD_PARAMETERS if 'Pb' in param}

    def _load_checkpoint(self):
            """Load saved mappings from checkpoint file"""
            if os.path.exists(self.checkpoint_path):
                try:
                    with open(self.checkpoint_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading checkpoint: {e}, starting fresh")
            return {}

    def _save_checkpoint(self):
        """Save current mappings to checkpoint file"""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.mappings, f, indent=2)

    def generate_mappings(self, input_dir):
        """Generate mappings for all files in directory"""
        files = [f for f in os.listdir(input_dir) if f.endswith('_Parameters.csv')]
        for fname in tqdm(files, desc="Processing parameter files"):
            base_name = fname.replace('_Parameters.csv', '')
            output_name = f"{base_name}.csv"

            # Skip if already processed
            if output_name in self.mappings:
                self.logger.info(f"Skipping already processed file: {fname}")
                continue

            file_map = self.process_file(os.path.join(input_dir, fname))

            if file_map:
                mapped_values = set(file_map.values())
                # Check if all geo parameters are present
                has_geo_params = self.geo_params.issubset(mapped_values)
                # Check if at least one Pb parameter is present
                has_pb_param = any(param in mapped_values for param in self.pb_params)

                if has_geo_params and has_pb_param:
                    self.mappings[output_name] = file_map
                    # Save after each file is processed
                    self._save_checkpoint()
                    self.logger.info(f"Processed and saved mapping for: {fname}")
                else:
                    missing = []
                    if not has_geo_params:
                        missing_geo = self.geo_params - mapped_values
                        missing.append(f"missing geo parameters: {', '.join(missing_geo)}")
                    if not has_pb_param:
                        missing.append("missing any Pb-related parameter")
                    self.logger.info(f"Skipping {fname}: {'; '.join(missing)}")
                
        return mappings

    def process_file(self, file_path):
        """Process a single parameters file"""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            self.logger.info(f"Error reading {file_path}: {e}")
            return None

        mapping = {}
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Mapping parameters"):
            prompt = PARAMETER_PROMPT_TEMPLATE.format(
                standard_params=self.standard_params,
                name=row['Supplied Name'],
                description=row.get('Supplied description', ''),
                unit=row.get('Supplied Units', '')
            )
            
            response = query_llm(prompt, self.model, self.tokenizer)
            matched_param = self._parse_response(response)
            
            if matched_param:
                mapping[row['Supplied Name']] = matched_param

        return mapping

    def _parse_response(self, response):
        """Extract parameter from LLM response"""
        try:
            param_part = response.split('<parameter>')[1].split('</parameter>')[0].strip()
            return param_part if param_part in STANDARD_PARAMETERS else None
        except (IndexError, AttributeError):
            return None