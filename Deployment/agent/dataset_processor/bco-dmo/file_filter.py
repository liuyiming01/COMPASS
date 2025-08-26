import os
import re
import json
from pathlib import Path
import pandas as pd

class FileFilter:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.param_pattern = re.compile(r'(\d+)_Parameters\.csv$')

    def _is_pb_related(self, param_file):
        try:
            df = pd.read_csv(param_file)
            return df['Supplied Name'].str.contains('Pb', case=False).any() or \
                   df['Supplied description'].str.contains('Pb', case=False).any()
        except Exception as e:
            return False

    def filter_files(self):
        pb_files = []
        for param_file in self.data_dir.glob('*_Parameters.csv'):
            if not self._is_pb_related(param_file):
                continue
            
            base_id = param_file.stem.split('_')[0]
            data_file = param_file.parent / f"{base_id}.csv"
            if data_file.exists():
                pb_files.append({
                    'data_file': str(data_file),
                    'param_file': str(param_file)
                })
        return pb_files