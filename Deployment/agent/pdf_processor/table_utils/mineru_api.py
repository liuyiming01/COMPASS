# Description: API for PDF processing using the Mineru library.
import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

def MinerU_PDF(pdf_path: str, output_dir: str) -> None:
    """
    Process a PDF file to extract and save its content and images.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory to save the output files.
    """
    # Prepare environment
    name_without_suffix = os.path.splitext(os.path.basename(pdf_path))[0]
    local_image_dir = os.path.join(output_dir, "images")
    local_md_dir = output_dir
    image_dir_name = os.path.basename(local_image_dir)

    os.makedirs(local_image_dir, exist_ok=True)

    # Initialize writers
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)

    # Read PDF bytes
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_path)

    # Create dataset instance
    ds = PymuDocDataset(pdf_bytes)

    # Determine processing method based on PDF classification
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # Extract content list and dump to JSON
    pipe_result.dump_content_list(md_writer, f"{name_without_suffix}_content_list.json", image_dir_name)

# Example usage:
# mineru_pdf("path/to/input.pdf", "path/to/output")