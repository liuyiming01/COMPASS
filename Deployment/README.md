# COMPASS Deployment

## Overview

**COMPASS Deployment** is the production implementation of our knowledge-tree enhanced LLM Agent framework for fine-grained scientific data integration. This deployment focuses on marine lead (Pb) and its isotope data extraction and integration from academic papers at scale.


## Installation

Clone the repository:

```bash
git clone https://github.com/liuyiming01/COMPASS.git
cd COMPASS/Deployment
```

Install the dependencies:

```bash
conda create -n compass-deploy python=3.10
conda activate compass-deploy
pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com
pip install vllm
```

## Quick Start

#### Basic Usage

```bash
bash scripts/dataset_process.sh
bash scripts/pdf_process.sh
bash scripts/table_process.sh
```

#### vLLM Inference Support

**MarinePbFusion** supports model inference using [vLLM](https://github.com/vllm-project/vllm), a high-performance LLM inference engine. For quantized inference, we recommend using **AWQ (Activation-aware Weight Quantization)** for higher generation quality and lower GPU memory usage.

1. Install vLLM:

   ```bash
   pip install vllm
   ```
2. Run inference with vLLM:

   ```bash
   export VLLM_WORKER_MULTIPROC_METHOD=spawn
   python agent/pdf_processor/main.py \
      --model_path Qwen/Qwen2.5-32B-Instruct-AWQ \
      --load_mode vllm \
      --vllm_quantization AWQ
   ```

   - `--load_mode vllm`: Enable vLLM for inference.
   - `--vllm_quantization AWQ`: Use AWQ quantization for efficient inference.