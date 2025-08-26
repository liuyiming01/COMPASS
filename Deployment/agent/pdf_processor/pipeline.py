# pdf_processor/processor_pipeline.py
import os
import csv
import json
import pandas as pd
from typing import Tuple, Dict, Callable
from pathlib import Path
import logging
from tqdm import tqdm

from .input_filters import raw_input_filter, classify_input_filter, marine_pb_filter
from .table_utils import PDFContentProcessor
from .utils import truncate_text, save_reorganize_response_to_csv


class ResearchPaperPipeline:
    def __init__(self, config: dict):
        self.prompts = config["prompts"]
        self.logger = logging.getLogger(__name__)
        self.step_filters: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
            "filter_pb_papers": raw_input_filter,
            "classify_pb_papers": classify_input_filter,
            "extract_pb_tables": marine_pb_filter,
            "extract_pb_tables_mineru": marine_pb_filter,
            "extract_pb_tables_query": marine_pb_filter,
            "reorganize_tables_data": marine_pb_filter,
        }
        self.logger.setLevel(logging.INFO)

    def process_csv_batch(
        self,
        step: str,
        input_csv: str,
        output_dir: str,
        llm_generator: Callable,
        batch_size: int = 4,
        pdf_dir: str = None
    ):
        """Generic CSV processing pipeline"""
        try:
            ## Load and filter data
            df = pd.read_csv(input_csv)
            filter_fn = self.step_filters.get(step, lambda df: df)
            df = filter_fn(df)

            ## Setup output
            output_path, result_file = self._setup_output(output_dir)

            # Handle PDF processing for extract_pb_tables
            if step == "extract_pb_tables" or step == "extract_pb_tables_mineru":
                pdf_processor = PDFContentProcessor(
                    input_df=df,
                    pdf_dir=pdf_dir,
                    mineru_output_dir=os.path.join(output_dir, "mineru_output"),
                )
                success, failure = pdf_processor.extract_pdf_content()
                content_list = pdf_processor.content_to_txt()
                if step == "extract_pb_tables_mineru":
                    return str(result_file)

            processed = self._get_processed_records(result_file)
            df_unprocessed = df[~df["_id"].isin(processed)]

            # Process in batches
            self._process_batches(df_unprocessed, step, output_dir, result_file, llm_generator, processed, batch_size)

            self.logger.info(f"Total records: {len(df)}; Already processed: {len(processed)}")
            self.logger.info(f"Successfully processed: {len(df_unprocessed)}")

            return str(result_file)
        except Exception as e:
            self.logger.error(f"CSV processing failed: {str(e)}")
            raise

    def _setup_output(self, output_dir: str) -> Tuple[Path, Path]:
        """Setup output directory and result file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / "results.csv"
        return output_path, result_file

    def _get_processed_records(self, result_file: Path) -> set:
        """Get already processed record IDs"""
        if not result_file.exists():
            return set()

        if not result_file.read_text():
            result_file.unlink()  # Delete empty file
            return set()
            
        df = pd.read_csv(result_file)
        return set(df["_id"].tolist())

    def _process_batches(self, df: pd.DataFrame, step: str, output_dir: str, 
                        result_file: Path, llm_generator: Callable, 
                        processed: set, batch_size: int):
        """Process dataframe in batches"""
        batch_size = max(1, batch_size)
        
        with result_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not processed:
                writer.writerow(df.columns.tolist() + [f"{step}_response"])

            for i in tqdm(range(0, len(df), batch_size), desc=f"Processing {step} batches"):
                batch = df.iloc[i:i+batch_size]
                batch_prompts = []
                batch_rows = []

                # Prepare prompts for batch
                for idx, row in batch.iterrows():
                    try:
                        prompt = self._prepare_prompt(row, step, output_dir)
                        batch_prompts.append(prompt)
                        batch_rows.append(row)
                    except Exception as e:
                        self.logger.error(f"Error processing {row['title']}: {str(e)}")

                # Process batch
                self._process_batch(batch_prompts, batch_rows, writer, llm_generator)

    def _prepare_prompt(self, row: pd.Series, step: str, output_dir: str) -> str:
        """Prepare prompt based on step type"""
        if step == "filter_pb_papers":
            return self.prompts["filter_pb_papers"].format(
                title=row["title"], 
                abstract=row["abstract"]
            )
        elif step == "classify_pb_papers":
            return self.prompts["classify_pb_papers"].format(
                title=row["title"], 
                abstract=row["abstract"]
            )
        elif step.startswith("extract_pb_tables"):
            pdf_id = os.path.splitext(os.path.basename(str(row["pdf_path"])))[0]
            content_list_with_table_json = os.path.join(output_dir, "mineru_output", pdf_id, "tables", 
                                            f"{pdf_id}_content_list_with_table.json")
            pdf_content, table_list = PDFContentProcessor.json_to_markdown(content_list_with_table_json)

            # truncated_content = truncate_text(pdf_content, max_length=20000)
            tables = "\n".join(table_list) if table_list else "No tables found."
            return self.prompts["extract_pb_tables"].format(
                title=row["title"], 
                abstract=row["abstract"],
                tables=tables
            )
        elif step == "reorganize_tables_data":
            pdf_id = os.path.splitext(os.path.basename(str(row["pdf_path"])))[0]
            content_list_with_table_json = os.path.join(os.path.dirname(output_dir), "3_extract_pb_tables", "mineru_output", pdf_id, "tables", 
                                            f"{pdf_id}_content_list_with_table.json")
            pdf_content, table_list = PDFContentProcessor.json_to_markdown(content_list_with_table_json)

            tables = "\n".join(table_list) if table_list else "No tables found."
            return self.prompts["reorganize_tables_data"].format(
                tables=tables
            )
        else:
            raise ValueError(f"Unknown step: {step}")

    def _process_batch(self, batch_prompts: list, batch_rows: list, 
                    writer: csv.writer, llm_generator: Callable):
        """Process a batch of prompts"""
        if not batch_prompts:
            return
        try:
            # Try batch processing
            batch_responses = llm_generator(batch_prompts)
            # Write results
            for row, response in zip(batch_rows, batch_responses):
                if response == "Generated error.":
                    continue  # Skip this response
                writer.writerow(row.tolist() + [response])

        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}, falling back to individual processing")
            # Fallback to individual processing
            for row, prompt in zip(batch_rows, batch_prompts):
                try:
                    response = llm_generator([prompt])[0]  # Process single item
                    if response == "Generated error.":
                        continue
                    writer.writerow(row.tolist() + [response])
                except Exception as inner_e:
                    self.logger.error(f"Error processing {row['title']}: {str(inner_e)}")


    def process(
        self,
        step: str,
        input_csv: str,
        output_dir: str,
        llm_generator: Callable = None,
        batch_size: int = 4,
        pdf_dir: str = None
    ):
        """Process input CSV file based on the specified step"""
        step_map = {
            "filter_pb_papers": 1,
            "classify_pb_papers": 2,
            "extract_pb_tables": 3,
            "extract_pb_tables_mineru": 3,
            "extract_pb_tables_query": 3,
            "reorganize_tables_data": 4
        }

        if step not in step_map:
            raise ValueError(f"Invalid processing step: {step}")
        self.logger.info(f"###### Processing step: {step} ######")

        step_num = step_map[step]
        if step.startswith("extract_pb_tables"):
            output_dir = os.path.join(output_dir, f"{step_num}_extract_pb_tables")
        else:
            output_dir = os.path.join(output_dir, f"{step_num}_{step}")

        result_file = self.process_csv_batch(
            step=step,
            input_csv=input_csv,
            output_dir=output_dir,
            llm_generator=llm_generator,
            batch_size=batch_size,
            pdf_dir=pdf_dir
        )

        if step == "reorganize_tables_data":
            tables_output_dir = os.path.join(output_dir, "reorganize_tables_csv")
            os.makedirs(tables_output_dir, exist_ok=True)
            save_reorganize_response_to_csv(result_file, tables_output_dir)

        return result_file