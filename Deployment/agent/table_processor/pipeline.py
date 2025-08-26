# table_processor/pipeline.py
import logging
import os
from typing import Callable, Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from .column_mapper import ColumnMapper
from .unit_converter import UnitConverter
from .utils import get_parameters, get_prompts

logger = logging.getLogger(__name__)

class TablePipeline:
    def __init__(self, strategy: str = 'rule_based'):
        self.strategy = strategy
        self.params = get_parameters()
        logger.info(f"Initialized table pipeline with strategy: {strategy}")

    def _setup_output(self, output_path: str) -> str:
        """Setup output directory and file paths"""
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _filter_valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        std_cols = self.params['standard_columns']
        mandatory_cols = self.params['mandatory_columns']

        filtered_df = pd.DataFrame()
        for col in std_cols:
            if col in df.columns:
                col_data = df.loc[:, col]
                if isinstance(col_data, pd.DataFrame):
                    # Merge duplicate columns by taking the first non-null value per row
                    merged = col_data.bfill(axis=1).iloc[:, 0]
                    filtered_df[col] = merged
                else:
                    filtered_df[col] = col_data
            else:
                filtered_df[col] = pd.NA

        # Remove rows with missing mandatory values
        filtered_df = filtered_df.dropna(subset=mandatory_cols)

        # Remove rows with data only in mandatory columns
        non_mandatory = [c for c in std_cols if c not in mandatory_cols]
        if non_mandatory:
            mask = filtered_df[non_mandatory].notna().any(axis=1)
            filtered_df = filtered_df[mask]

        # Remove duplicate rows
        filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)
    
        # 按规则做范围过滤
        ranges: Dict[str, tuple] = {
            'Pb_CONC [pmol/kg]': (0, 1000),
            'Pb_210_CONC [mBq/kg]': (0, 10),
            'Pb_206_204': (0, 100),
            'Pb_206_207': (0, 100),
            'Pb_208_206': (0, 100),
            'Pb_207_204': (0, 100),
            'Pb_208_207': (0, 100),
            'Pb_208_204': (0, 100),
        }
        for col, (min_v, max_v) in ranges.items():
            if col in filtered_df.columns:
                # 转数值、coerce 非数字为 NaN
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                # 保留在区间内或 NaN
                mask = filtered_df[col].between(min_v, max_v) | filtered_df[col].isna()
                filtered_df = filtered_df[mask]

        filtered_df = filtered_df.reset_index(drop=True)
        return filtered_df

    def _process_single_file(
        self,
        csv_path: str,
        output_dir: str,
        llm_generator: Optional[Callable],
        batch_size: int
    ) -> Optional[pd.DataFrame]:
        """Process a single CSV file through standardization pipeline."""
        logger.info(f"Processing file: {os.path.basename(csv_path)}")
        
        # Column standardization
        column_processor = ColumnMapper(
            strategy=self.strategy, 
            llm_generator=llm_generator
        )
        column_mapping = column_processor.standardize(csv_path, output_dir)

        # Unit conversion
        unit_converter = UnitConverter(llm_generator=llm_generator)
        converted_df = unit_converter.convert_units(
            csv_path, output_dir, column_mapping
        )

        filtered_df = self._filter_valid_rows(converted_df)
        return filtered_df

    def process(
        self,
        input_dir: str,
        output_path: str,
        llm_generator: Callable = None,
        batch_size: int = 4,
    ):
        output_dir = self._setup_output(output_path)
        csv_files = [os.path.join(input_dir, f) 
                    for f in os.listdir(input_dir) if f.endswith('.csv')]

        processed_data = []
        with tqdm(csv_files, desc="Processing files") as progress_bar:
            for csv_file in progress_bar:
                df = self._process_single_file(
                    csv_file, output_dir, llm_generator, batch_size
                )
                if df is not None and not df.empty:
                    df.insert(0, 'source_file', os.path.basename(csv_file))
                    processed_data.append(df)

        if not processed_data:
            logger.warning("No valid files processed")
            return

        final_df = pd.concat(processed_data, ignore_index=True)
        final_df.to_csv(output_path, index=False)
        logger.info(f"Pipeline completed. Output saved to {output_path}")