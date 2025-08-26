# pdf_processor/table_utils/pdf_content_processor.py
import os
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
import logging
from .mineru_api import MinerU_PDF
from .table_image_processor import process_table_image_pipeline


class PDFContentProcessor:
    def __init__(
        self,
        input_df: pd.DataFrame,
        pdf_dir: str,
        mineru_output_dir: str,
        mineru_mode: str = "auto",
    ):
        """Initialize PDF content processor"""
        self.logger = logging.getLogger(__name__)
        self.input_df = input_df

        if pdf_dir and not os.path.exists(pdf_dir):
            raise ValueError(f"PDF directory not found: {pdf_dir}")
        self.pdf_dir = pdf_dir
        self.mineru_output_dir = mineru_output_dir
        self.mineru_mode = mineru_mode

        self._create_directories()

    def _create_directories(self):
        """Ensure all required directories exist"""
        os.makedirs(self.mineru_output_dir, exist_ok=True)

    def extract_pdf_content(self) -> Tuple[int, int]:
        """Extract content from PDFs using MinerU"""
        unprocessed = self._get_unprocessed_pdfs()
        self.logger.info(f"Starting MinerU PDF processing. Total: {len(unprocessed)}")
        success, failure = 0, 0
        for pdf_path in tqdm(unprocessed, desc="Processing PDFs in MinerU", unit="PDF"):
            if self.pdf_dir:
                local_pdf_path = os.path.join(self.pdf_dir, os.path.basename(pdf_path))
            else:
                local_pdf_path = pdf_path
            try:
                if not os.path.exists(local_pdf_path):
                    self.logger.error(f"PDF file not found: {local_pdf_path}")
                    continue
                MinerU_PDF(local_pdf_path, os.path.join(self.mineru_output_dir, os.path.splitext(os.path.basename(local_pdf_path))[0]))
                success += 1
            except Exception as e:
                self.logger.error(f"Unexpected error with {local_pdf_path}: {str(e)}")
                failure += 1

        self.logger.info(f"MinerU PDF extraction completed. Success: {success}, Failures: {failure}")
        return success, failure

    def _get_unprocessed_pdfs(self) -> list:
        """Identify unprocessed PDF files"""
        try:
            df = self.input_df
            if 'pdf_path' not in df.columns:
                raise ValueError("CSV file must contain a 'pdf_path' column")
            all_pdfs = [
                str(path) for path in df['pdf_path'].tolist()
                if pd.notna(path) and str(path).lower().endswith('.pdf')
            ]
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        processed = set()
        for name in os.listdir(self.mineru_output_dir):
            dir_path = os.path.join(self.mineru_output_dir, name)
            if os.path.isdir(dir_path):
                if any(f.endswith("_content_list.json") for f in os.listdir(dir_path)):
                    processed.add(name.lower())

        return [
            f for f in all_pdfs
            if os.path.splitext(os.path.basename(f))[0].lower() not in processed
        ]

    def content_to_txt(self):
        """Finish the PDF table convertion to text format"""
        pdf_ids = sorted([
            pid for pid in os.listdir(self.mineru_output_dir)
            if os.path.isdir(os.path.join(self.mineru_output_dir, pid))
        ])

        for pdf_id in tqdm(pdf_ids, desc="Processing PDFs", unit="PDF"):
            try:
                self._process_single_pdf(pdf_id)
            except Exception as e:
                self.logger.error(f"Error processing {pdf_id}: {str(e)}")

        return sorted([
            os.path.join(self.mineru_output_dir, pdf_id, "tables", f"{pdf_id}_content_list_with_table.json")
            for pdf_id in pdf_ids
        ])

    def _process_single_pdf(self, pdf_id: str):
        """Process content conversion for a single PDF"""
        # source_dir = os.path.join(self.mineru_output_dir, pdf_id, self.mineru_mode)
        source_dir = os.path.join(self.mineru_output_dir, pdf_id)
        target_dir = os.path.join(self.mineru_output_dir, pdf_id, "tables")
        
        if self._is_already_processed(target_dir):
            return

        content_file = os.path.join(source_dir, f"{pdf_id}_content_list.json")
        if not os.path.exists(content_file):
            return

        with open(content_file, "r", encoding="utf-8") as f:
            content_data = json.load(f)

        self._process_tables(content_data, source_dir, target_dir)

        os.makedirs(target_dir, exist_ok=True)
        output_path = Path(target_dir) / f"{pdf_id}_content_list_with_table.json"
        with output_path.open("w", encoding='utf-8') as f:
            json.dump(content_data, f, indent=4, ensure_ascii=False)

    def _is_already_processed(self, target_dir: str) -> bool:
        """Check if PDF has already been processed"""
        return os.path.exists(target_dir) and any(
            f.endswith("_content_list_with_table.json") for f in os.listdir(target_dir)
        )

    def _process_tables(self, content_data: list, source_dir: str, target_dir: str):
        """Process table images and update content data"""
        table_images = [
            os.path.join(source_dir, item["img_path"].strip())
            for item in content_data
            if item.get("type") == "table" and item.get("img_path")
        ]

        image_output = os.path.join(target_dir, "images")
        os.makedirs(image_output, exist_ok=True)
        
        # Convert the table image to a markdown format table.
        table_bodies = process_table_image_pipeline(table_images, image_output)
        
        # Update table content in-place
        table_idx = 0
        for item in content_data:
            if item.get("type") == "table":
                item["table_body"] = table_bodies[table_idx] if table_idx < len(table_bodies) else ""
                table_idx += 1


    @staticmethod
    def json_to_markdown(json_path: str) -> Tuple[str, List[str]]:
        """Convert JSON content to formatted markdown for LLM input"""
        def safe_json_load(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"JSON load failed: {str(e)}")
                return []

        content = []
        tables = []
        
        for item in safe_json_load(json_path):
            if item["type"] == "text":
                content.append(PDFContentProcessor._format_text(item))
            elif item["type"] == "table":
                table_str = PDFContentProcessor._format_table(item)
                if table_str:
                    tables.append(table_str)
        
        return "\n\n".join(content), tables

    @staticmethod
    def _format_text(item: dict) -> str:
        """Format text elements with heading levels"""
        level = item.get("text_level")
        text = item.get("text", "")
        return f"{'#' * level} {text}" if level else text

    @staticmethod
    def _format_table(item: dict) -> str:
        """Format table elements with caption and footnotes"""
        caption = "".join(item.get("table_caption", []))
        body = item.get("table_body", "")
        footnote = "".join(item.get("table_footnote", []))
        
        if not any([caption, body, footnote]):
            return ""
            
        return rf"{caption}\n{body}\n{footnote}".strip()