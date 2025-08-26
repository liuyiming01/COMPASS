import os
import json
from pathlib import Path
from mineru_api import MinerU_PDF
from table_image_processor import process_table_image_pipeline


def _process_single_pdf(mineru_output_dir, pdf_id):
        """Process content conversion for a single PDF"""
        # source_dir = os.path.join(self.mineru_output_dir, pdf_id, self.mineru_mode)
        source_dir = os.path.join(mineru_output_dir, pdf_id)
        target_dir = os.path.join(mineru_output_dir, pdf_id, "tables")


        content_file = os.path.join(source_dir, f"{pdf_id}_content_list.json")
        if not os.path.exists(content_file):
            return

        with open(content_file, "r", encoding="utf-8") as f:
            content_data = json.load(f)

        _process_tables(content_data, source_dir, target_dir)

        os.makedirs(target_dir, exist_ok=True)
        output_path = Path(target_dir) / f"{pdf_id}_content_list_with_table.json"
        with output_path.open("w", encoding='utf-8') as f:
            json.dump(content_data, f, indent=4, ensure_ascii=False)

def _process_tables(content_data: list, source_dir: str, target_dir: str):
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



local_pdf_path = "/home/lym/PbCirculation/MarinePbFusion/agent/pdf_processor/outputs/0315test/marine_pdfs_20250315_185530/0103-5053.20180228.pdf"
mineru_output_dir = "/home/lym/PbCirculation/MarinePbFusion/agent/pdf_processor/outputs/0318test/mineru"

os.makedirs(mineru_output_dir, exist_ok=True)
# MinerU_PDF(local_pdf_path, os.path.join(mineru_output_dir, os.path.splitext(os.path.basename(local_pdf_path))[0]))

_process_single_pdf(mineru_output_dir, "0103-5053.20180228")