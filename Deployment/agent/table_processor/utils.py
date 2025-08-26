# table_processor/utils.py
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import re

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def get_parameters(params_path: str = None) -> Dict[str, Any]:
    params_path = Path(__file__).parent.parent / "config/table_config/parameters.yaml" if params_path is None else Path(params_path)
    return load_config(params_path)

def get_prompts(prompts_path: str = None) -> Dict[str, Any]:
    prompts_path = Path(__file__).parent.parent / "config/table_config/prompts.yaml" if prompts_path is None else Path(prompts_path)
    return load_config(prompts_path)

def load_config(file_path: Path) -> Dict[str, Any]:
    """Load config from YAML file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_unit_from_column(column_name: str) -> str:
    """Extract unit from the last occurrence of [unit] or (unit) in column name"""
    bracket_matches = re.findall(r'\[([^\]]+)\]|\(([^\)]+)\)', column_name)
    if not bracket_matches:
        return None
    
    last_match = bracket_matches[-1]
    return last_match[0] if last_match[0] else last_match[1]

def remove_brackets(text: str) -> str:
    """Remove only the last occurrence of a bracket (parentheses or square brackets) and its content from a string."""
    pattern = r'(\([^)]*\)|\[[^]]*\])'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text.strip()
    last_match = matches[-1]
    result = text[:last_match.start()] + text[last_match.end():]
    return result.strip()