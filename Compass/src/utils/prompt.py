import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional


# Mapping from original label to simplified label.
ORIGINAL_TO_SIMPLIFIED = {
    "Pb_data_papers": "Marine_Pb",
    "Pb210_data_papers": "Marine_Pb", 
    "Pb_ratio_data_papers": "Marine_Pb",
    "marine_Pb_no_data": "Marine_Pb",
    "atmospheric_Pb_papers": "Atmospheric_Pb",
    "terrestrial_Pb_papers": "Terrestrial_Pb",
    "chemical_Pb_papers": "Other",
    "marine_element_papers": "Other",
    "unrelated_papers_Pb_keywords": "Other",
    "unrelated_papers": "Other"
}


def convert_to_simplified_labels(original_labels: List[str]) -> List[str]:
    """Convert original labels to simplified labels"""
    return [ORIGINAL_TO_SIMPLIFIED.get(label, "Other") for label in original_labels]

def extract_prediction_from_response(response: str) -> Optional[str]:
    """Extract predicted label from LLM response using strict parsing"""
    if not response:
        return None

    response = response.strip()
    tag_pattern = r'<answer>\s*([^<>]+)\s*</answer>'
    tag_match = re.search(tag_pattern, response, re.IGNORECASE)
    if tag_match:
        prediction = tag_match.group(1).strip()
        return prediction

    return None
    
def load_prompts_from_yaml(yaml_path: str = None) -> Dict:
    """Load prompts from YAML config file"""
    if yaml_path is None:
        current_dir = Path(__file__).parent
        yaml_path = current_dir.parent / "config" / "prompts2.0.yaml"

    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        return config.get('prompts', {})

def generate_prompt(
    task: str,
    title: str = None,
    abstract: str = None,
    table_content: str = None,
    tables: str = None,
    yaml_path: str = None
) -> str:
    """
    Unified prompt generator.
    task: 
        'PaperClassification', 
        'TableClassification', 
        'E2EDataExtraction', 
        'knowledge_tree_PaperClassification', 
        'knowledge_tree_TableClassification', 
        'knowledge_tree_E2EDataExtraction',
    """
    prompts = load_prompts_from_yaml(yaml_path)
    knowledge_tree_prompts = prompts.get('knowledge_tree_prompt', {})

    if task.startswith("knowledge_tree_"):
        if task == 'knowledge_tree_PaperClassification':
            prompt_template1 = knowledge_tree_prompts.get('Pb_Related_Paper', '')
            prompt_template2 = knowledge_tree_prompts.get('PaperClassification', '')
            return prompt_template1.format(title=title, abstract=abstract), prompt_template2.format(title=title, abstract=abstract)
        elif task == 'knowledge_tree_TableClassification':
            prompt_template1 = knowledge_tree_prompts.get('Pb_Related_Table', '')
            prompt_template2 = knowledge_tree_prompts.get('TableClassification', '')
            return prompt_template1.format(title=title, abstract=abstract, table_content=table_content), prompt_template2.format(title=title, abstract=abstract, table_content=table_content)
        elif task == 'knowledge_tree_E2EDataExtraction':
            prompt_template = knowledge_tree_prompts.get('E2EDataExtraction', '')
            return prompt_template.format(tables=tables)
    else:
        prompt_template = prompts.get(task, '')

        if task == 'PaperClassification':
            return prompt_template.format(title=title, abstract=abstract)
        elif task == 'TableClassification':
            return prompt_template.format(title=title, abstract=abstract, table_content=table_content)
        elif task == 'E2EDataExtraction':
            return prompt_template.format(title=title, abstract=abstract, tables=tables)
        elif task == 'E2ETableClassification':
            prompt_template = knowledge_tree_prompts.get('E2ETableClassification', '')
            return prompt_template.format(title=title, abstract=abstract, table_content=table_content)
        else:
            raise ValueError(f"Unknown task: {task}")

def generate_ablation_prompt(
    task: str,
    title: str = None,
    abstract: str = None,
    table_content: str = None,
    tables: str = None,
    yaml_path: str = None
) -> str:
    """
    Unified prompt generator.
    task: 
        'wo_knowledge_node_PaperClassification', 
        'wo_knowledge_node_TableClassification', 
        'wo_knowledge_node_E2EDataExtraction',

        'wo_tree_struct_PaperClassification', 
        'wo_tree_struct_TableClassification', 
        'wo_tree_struct_E2EDataExtraction',
    """
    prompts = load_prompts_from_yaml("/home/lym/PbCirculation/Compass/code/config/ablation_prompts.yaml")

    if task.startswith("wo_knowledge_node"):
        wo_knowledge_node_prompts = prompts.get('wo_knowledge_node', {})
        if task == 'wo_knowledge_node_PaperClassification':
            prompt_template1 = wo_knowledge_node_prompts.get('Pb_Related_Paper', '')
            prompt_template2 = wo_knowledge_node_prompts.get('PaperClassification', '')
            return prompt_template1.format(title=title, abstract=abstract), prompt_template2.format(title=title, abstract=abstract)
        elif task == 'wo_knowledge_node_TableClassification':
            prompt_template1 = wo_knowledge_node_prompts.get('Pb_Related_Table', '')
            prompt_template2 = wo_knowledge_node_prompts.get('TableClassification', '')
            return prompt_template1.format(title=title, abstract=abstract, table_content=table_content), prompt_template2.format(title=title, abstract=abstract, table_content=table_content)
        elif task == 'wo_knowledge_node_E2EDataExtraction':
            prompt_template = wo_knowledge_node_prompts.get('E2EDataExtraction', '')
            return prompt_template.format(tables=tables)

    elif task.startswith("wo_tree_struct"):
        wo_tree_struct_prompts = prompts.get("wo_tree_struct", {})
        if task == 'wo_tree_struct_PaperClassification':
            prompt_template = wo_tree_struct_prompts.get('PaperClassification', '')
            return prompt_template.format(title=title, abstract=abstract)
        elif task == 'wo_tree_struct_TableClassification':
            prompt_template = wo_tree_struct_prompts.get('TableClassification', '')
            return prompt_template.format(title=title, abstract=abstract, table_content=table_content)
        elif task == 'wo_tree_struct_E2EDataExtraction':
            prompt_template = wo_tree_struct_prompts.get('E2EDataExtraction', '')
            return prompt_template.format(title=title, abstract=abstract, tables=tables)
    else:
        raise ValueError(f"Unknown task: {task}")
