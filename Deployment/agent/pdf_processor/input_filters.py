# pdf_processor/filters.py

import pandas as pd
import re
import logging


def raw_input_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter rows where 'title' and 'abstract' is missing."""
    return df.dropna(subset=["title", "abstract"], how="all")

def extract_answer(response: str) -> str:
    """
    Extract the content between <answer> and </answer> tags.
    If no valid tag is found, return an empty string.
    """
    match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def classify_input_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows based on the answer between <answer> and </answer> tags.
    Only include rows where the answer is 'Yes'.
    """
    df = df.dropna(subset=["filter_pb_papers_response"])

    yes_mask = df["filter_pb_papers_response"].apply(extract_answer).str.lower() == "yes"
    no_mask = df["filter_pb_papers_response"].apply(extract_answer).str.lower() == "no"
    
    yes_df = df[yes_mask]
    no_df = df[no_mask]

    stats = (
        f"\nYes responses: {len(yes_df)}/{len(df)}\n"
        f"No responses: {len(no_df)}/{len(df)}\n"
    )
    logging.info(stats)

    return yes_df

def marine_pb_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows based on the answer between <answer> and </answer> tags.
    Only include rows where the answer is 'Marine'.
    """
    df = df.dropna(subset=["classify_pb_papers_response"])

    extracted_answers = df["classify_pb_papers_response"].apply(extract_answer).str.lower()

    marine_df = df[extracted_answers == "marine"]
    terrestrial_df = df[extracted_answers == "terrestrial"]
    atmospheric_df = df[extracted_answers == "atmospheric"]
    interface_df = df[extracted_answers == "interface regions"]
    other_df = df[~extracted_answers.isin(["marine", "terrestrial", "atmospheric", "interface regions"])]

    stats = (
        f"\nMarine Category: {len(marine_df)}/{len(df)}\n"
        f"Terrestrial Category: {len(terrestrial_df)}/{len(df)}\n"
        f"Atmospheric Category: {len(atmospheric_df)}/{len(df)}\n"
        f"Interface Regions Category: {len(interface_df)}/{len(df)}\n"
        f"Other Category: {len(other_df)}/{len(df)}\n"
    )
    logging.info(stats)

    # Filter out rows without PDF paths
    marine_df = marine_df[
        marine_df["pdf_path"].notna() & marine_df["pdf_path"].str.endswith('.pdf') 
    ]
    return marine_df



"""Input filter functions for the old version prompts."""
# def _raw_input_filter(self, df: pd.DataFrame) -> pd.DataFrame:
#     return df.dropna(subset=["title", "abstract"])

# def _classify_input_filter(self, df: pd.DataFrame) -> pd.DataFrame:
#     """Filter rows based on LLM responses starting with 'yes' and log response statistics."""
#     df = df.dropna(subset=["filter_pb_papers_response"])
#     responses = df["filter_pb_papers_response"].str.lower()

#     yes_df = df[responses.str.startswith("yes")]
#     no_df = df[responses.str.startswith("no")]

#     stats = (
#         f"\nYes responses: {len(yes_df)}/{len(df)}\n"
#         f"No responses: {len(no_df)}/{len(df)}\n"
#     )
#     self.logger.info(stats)

#     return yes_df

# def marine_pb_filter(df: pd.DataFrame) -> pd.DataFrame:
#     """Filter rows based on LLM responses that indicate 'category: marine'."""
#     df = df.dropna(subset=["classify_pb_papers_response"])
#     responses = df["classify_pb_papers_response"].str.lower()

#     marine_df = df[responses.str.startswith('category: marine')]
#     terrestrial_df = df[responses.str.startswith('category: terrestrial')]
#     atmospheric_df = df[responses.str.startswith('category: atmospheric')]
#     interface_df = df[responses.str.startswith('category: interface regions')]
#     other_df = df[~responses.str.startswith('category: marine') & ~responses.str.startswith('category: terrestrial') & ~responses.str.startswith('category: atmospheric') & ~responses.str.startswith('category: interface regions')]

#     stats = (
#         f"\nMarine Category: {len(marine_df)}/{len(df)}\n"
#         f"Terrestrial Category: {len(terrestrial_df)}/{len(df)}\n"
#         f"Atmospheric Category: {len(atmospheric_df)}/{len(df)}\n"
#         f"Interface Regions Category: {len(interface_df)}/{len(df)}\n"
#         f"Other Category: {len(other_df)}/{len(df)}\n"
#     )
#     logging.info(stats)

#     # Filter out rows without PDF paths
#     marine_df = marine_df[
#         marine_df["pdf_path"].notna() & marine_df["pdf_path"].str.endswith('.pdf') 
#     ]

#     return marine_df