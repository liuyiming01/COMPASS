import os
import pandas as pd

csv_paths = [
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/Pb_data_papers_16.csv",
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/Pb210_data_papers_23.csv",
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/Pb_ratio_data_papers_10.csv",
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/marine_Pb_no_data_13.csv",
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/atmospheric_Pb_papers_16.csv",
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/terrestrial_Pb_papers_13.csv",
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/chemical_Pb_papers_11.csv",
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/marine_element_papers_18.csv",
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/unrelated_papers_Pb_keywords_11.csv",
    "/home/lym/PbCirculation/Compass/test_data/selected_data/checked/unrelated_papers_18.csv"
]
labels = [
    "Pb_data_papers",
    "Pb210_data_papers", 
    "Pb_ratio_data_papers",
    "marine_Pb_no_data",
    "atmospheric_Pb_papers",
    "terrestrial_Pb_papers",
    "chemical_Pb_papers",
    "marine_element_papers",
    "unrelated_papers_Pb_keywords",
    "unrelated_papers"
]

def Dataloader(test_data_path=None):
    if test_data_path is None:
        test_data_path = "/home/lym/PbCirculation/Compass/test_data/test_data2.csv"
    df = pd.read_csv(test_data_path)
    return df
