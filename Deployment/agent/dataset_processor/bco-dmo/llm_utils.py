# llm_utils.py
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def initialize_model(model_path):
    """Initialize LLM model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def query_llm(prompt, model, tokenizer):
    """Execute LLM query with structured prompt"""
    messages = [{
        "role": "system", 
        "content": "You are an oceanographic data standardization expert."
    }, {
        "role": "user", 
        "content": prompt
    }]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_conversion_formula(original_unit, standard_unit, header_name):

    prompt = f"""
    Please write a Python function to convert a numerical value from the specified original unit to the standard unit for seawater lead element data. The function should handle straightforward unit conversions and return None if conversion isn't possible due to incompatible units.

    Function Requirements:
    - Input: A float representing the value in original_unit.
    - Output: A float representing the converted value in standard_unit, or `None` if conversion isn't possible.

    Instructions:
    1. Provide conversion logic if units can be directly converted using a mathematical factor.
    2. Return None if units have incompatible dimensions or no conversion factor exists.
    3. Consider standard unit prefixes (nano, micro, milli, etc.).
    4. When necessary, account for the density of seawater to convert between mass and volume units (assume a standard seawater density of 1.025 kg/L).
    5. Use Lead's molar mass (207.2 g/mol) for conversions involving moles.
    6. Handle unit representations with exponents (e.g., "m³" as cubic meters, "100L-1" as per 100 liters).
    7. Enclose the function within <func> and </func> blocks.

    Examples:

    Example 1:
    - Original Unit: Bq/kg
    - Standard Unit: mBq/kg
    - Conversion Logic: Multiply by 1000 to convert Bq to mBq.
    <func>
    def convert_lead_unit(value):
        return value * 1000
    </func>

    Example 2:
    - Original Unit: nmol/kg
    - Standard Unit: pmol/kg
    - Conversion Logic: Multiply by 1000 to convert nmol to pmol.
    <func>
    def convert_lead_unit(value):
        return value * 1000
    </func>

    Example 3:
    - Original Unit: dpm/gram
    - Standard Unit: mBq/kg
    - Conversion Logic: gram→kg: (×1000); dpm→Bq: (/60); Bq→mBq: (×1000)
    <func>
    def convert_lead_unit(value):
        return value * 1000 / 60 * 1000
    </func>

    Example 4:
    - Original Unit: 210Pb dpm 100L-1
    - Standard Unit: mBq/kg
    - Conversion Logic: 100L seawater = 102.5 kg; dpm→Bq: (/60); Bq→mBq: (×1000)
    <func>
    def convert_lead_unit(value):
        density_kg_per_100L = 100 * 1.025
        return value / density_kg_per_100L / 60 * 1000
    </func>

    Example 5:
    - Original Unit: ng/m³
    - Standard Unit: pmol/kg
    - Conversion Logic: m³→kg using density (÷1025 kg/m³), ng→g (×1e-9), g→mol (÷207.2), mol→pmol (×1e12).
    <func>
    def convert_lead_unit(value):
        density_kg_per_m3 = 1025  # 1.025 kg/L * 1000 L/m³
        return value / density_kg_per_m3 * 1e-9 / 207.2 * 1e12
    </func>

    ---

    Task:
    - Original Unit: {original_unit}
    - Standard Unit: {standard_unit}
    """
    return prompt

def extract_convert_function(model_output):
    """
    Extracts the Python 'convert' function from the model output.
    Handles two cases:
    1. Direct Python code between <func> and </func>
    2. Python code enclosed within ```python and ```

    Parameters:
        model_output (str): The string output from the model.

    Returns:
        function or None: The extracted 'convert' function if successful, otherwise None.
    """
    pattern = r"<func>(.*?)</func>"
    match = re.search(pattern, model_output, re.DOTALL)
    if match:
        func_code = match.group(1).strip()

        # Check if the extracted code is enclosed within triple backticks
        if func_code.startswith("```"):
            # Remove the opening ```python or ``` if present
            func_code = re.sub(r'^```(?:python)?\s*', '', func_code, flags=re.IGNORECASE).strip()
            # Remove the closing ```
            func_code = re.sub(r'\s*```$', '', func_code).strip()

        local_namespace = {}
        try:
            exec(func_code, {}, local_namespace)
            return local_namespace.get('convert')
        except Exception as e:
            print(f"Error executing convert function code: {e}")
            return None
    else:
        print("No <func> tag found in the model output.")
    return None
