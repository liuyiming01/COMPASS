# pdf_processor/llm_loader.py
# Description: Load LLM models using transformers or vLLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional, Union, List

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class LLMHandler:
    def __init__(
        self, 
        model_path: str,
        load_mode: str = "transformers",
        trust_remote_code: bool = True,
        # transformers params
        quantization_config: Optional[BitsAndBytesConfig] = None,
        # vLLM params
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        vllm_quantization: str = "bitsandbytes"
    ):

        self.load_mode = load_mode
        
        if load_mode == "transformers":
            self.model, self.tokenizer = self._load_transformers_model(
                model_path,
                trust_remote_code=trust_remote_code,
                quantization_config=quantization_config
            )
        elif load_mode == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is not available. Please install with 'pip install vllm'")
            self.model, self.tokenizer = self._load_vllm_model(
                model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                quantization=vllm_quantization,
                trust_remote_code=trust_remote_code
            )
        else:
            raise ValueError(f"Unsupported load mode: {load_mode}")

    def _load_transformers_model(
        self,
        model_path: str,
        trust_remote_code: bool,
        quantization_config: Optional[BitsAndBytesConfig] = None
    ):
        """Load transformers LLM model"""
        if quantization_config is None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        return model, tokenizer

    def _load_vllm_model(
        self,
        model_path: str,
        tensor_parallel_size: Optional[int],
        gpu_memory_utilization: float,
        max_model_len: int = 8192,
        quantization: str = "bitsandbytes",
        trust_remote_code: bool = True
    ):
        """Load vLLM model"""
        if quantization == "bitsandbytes":
            model = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                # dtype=torch.bfloat16,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
            )
        elif quantization == "AWQ":
            model = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
                quantization="AWQ",
                max_model_len=max_model_len
            )
        else:
            raise ValueError(f"Unsupported quantization: {quantization}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        return model, tokenizer

    def create_generator(
        self,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        seed: int = 42,
        deterministic: bool = True
    ):
        """Create text generation pipeline"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
        if self.load_mode == "transformers":
            def generate(prompts: Union[str, List[str]]) -> Union[str, List[str]]:
                if isinstance(prompts, str):
                    prompts = [prompts]
                try:
                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                    ).to(self.model.device)

                    outputs = self.model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    results = []
                    for i, output_ids in enumerate(outputs):
                        full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                        generated_part = full_text[len(prompts[i]):].strip()
                        results.append(generated_part)

                    return results
                except Exception as e:
                    print(f"Error during generation: {e}")
                    return ["Generated error."] * len(prompts)

            return generate

        elif self.load_mode == "vllm":
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                stop_token_ids=[self.tokenizer.eos_token_id],
                # truncate_prompt_tokens=10000
            )

            def generate(prompts: Union[str, List[str]]) -> Union[str, List[str]]:
                if isinstance(prompts, str):
                    prompts = [prompts]
                try:
                    outputs = self.model.generate(prompts, sampling_params)
                    results = [out.outputs[0].text.strip() for out in outputs]
                    return results
                except Exception as e:
                    print(f"Error during generation: {e}")
                    return ["Generated error."] * len(prompts)

            return generate

        raise ValueError(f"Unsupported load mode: {self.load_mode}")