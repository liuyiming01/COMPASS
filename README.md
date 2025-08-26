# COMPASS: Scientific LLM Agent for Fine-grained Data Integration in Academic Papers

<div align="center">

[![Website](https://img.shields.io/badge/Website-Visit%20Here-blue)](https://jingwei.acemap.info/lead)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

## Overview

**COMPASS** is a knowledge-tree enhanced LLM Agent framework designed for fine-grained scientific data integration from academic papers. It bridges the domain gap between general-purpose language models and specialized scientific domains through low-cost expert-guided adaptation, enabling reliable extraction and integration of observational data from diverse academic sources.

COMPASS hierarchically decomposes data integration tasks into manageable subtasks through a three-phase workflow:

1. **Collection Phase**: Identifies relevant papers and data sources
2. **Extraction Phase**: Extracts structured data from multimodal content
3. **Aggregation Phase**: Harmonizes and integrates extracted data

## Real-world Impact

In collaboration with marine scientists, COMPASS has been successfully deployed to conduct data integration on global marine lead (Pb) and its isotopes:

- ✅ **100+ relevant papers** identified from over 230,000 open-access academic papers
- ✅ **3,751 new Pb records** extracted and integrated
- ✅ **Interactive visualization platform** launched at [https://jingwei.acemap.info/lead](https://jingwei.acemap.info/lead)
- ✅ **New data insights** for regions including the East China Sea and Southern Ocean

## Getting Started

### Prerequisites

- Python 3.8+
- Required dependencies (see `requirements.txt`)

### Installation

```bash
git clone https://github.com/liuyiming01/COMPASS.git
cd COMPASS
```

### Quick Start
1. Testing COMPASS Method:
```bash
cd Compass/evaluation/eval/Qwen2.5
bash eval.sh
```

2. Running Baseline Comparisons:
```bash
cd Compass/evaluation/baselines/Qwen/Qwen2.5
bash eval.sh
```

## Data and Evaluation

### Test Data

Due to copyright protection, we provide only DOI numbers for test papers and result papers, without PDF files or other copyrighted materials. The evaluation framework includes:

- Baseline comparisons with GPT-4o and domain-specific LLMs
- Ablation studies on key components
- Performance metrics for accuracy and reliability


## Online Platform

Visit our interactive visualization platform to explore the integrated marine lead data:

🌐 **[https://jingwei.acemap.info/lead](https://jingwei.acemap.info/lead)**