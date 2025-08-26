# config.py
STANDARD_PARAMETERS = [
    'Longitude [degrees_east]',
    'Latitude [degrees_north]',
    'DEPTH [m]',
    'Pb_CONC [pmol/kg]',
    'Pb_210_CONC [mBq/kg]',
    'Pb_206_204',
    'Pb_206_207',
    'Pb_208_206',
    'Pb_207_204',
    'Pb_208_207',
    'Pb_208_204'
]

PARAMETER_PROMPT_TEMPLATE = """Analyze the provided oceanographic data column metadata and match it to standardized parameters. Respond ONLY with the exact parameter name from the list below or 'None':

Standard Parameters:
{standard_params}

Column Metadata:
- Name: {name}
- Description: {description}
- Unit: {unit}

Required response format:

<details> Brief reasoning </details>
<answer>Matched parameter name or None</answer>"""