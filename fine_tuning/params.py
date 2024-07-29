from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    train_file: str = field(
        default=None, 
        metadata={"help": "Path to the train data in jsonl format."}
    )
    dataset_text_field: Optional[str] = field(
        default="instruction", 
        metadata={"help": "The field in dataset for completion pretraining. Mandatory if 'pre_sft' is True."}
    )
    user_prompt_format: Optional[str] = field(
        default="llama2",
        metadata={"help": "The name of a known llm model prompt format or a the user custom prompt. Mandatory if 'instruct_sft' is True."}
    )
    user_response_sentence: Optional[str] = field(
        default=None,
        metadata={"help": "The response sentence for instruction pretraining"}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )
    adapter_paths: List[str] = field(
        default=None,
        metadata={"help": "A list of paths to multi model adapters."},
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Max length of the tokenizer"},
    )
    huggingface_token: Optional[str] = field(
        default=None,
        metadata={"help": "Huggingface token for private model"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use fast tokenizer"},
    )
    output_dir: str = field(
        default="./",
        metadata={"help": "Output directory for the model"},
    )


@dataclass
class TrainingArguments:
    pre_sft: bool = field(
        default=False,
        metadata={"help": "Whether to pretrain the model"},
    )
    instruct_sft: bool = field(
        default=True,
        metadata={"help": "Whether to instruct the model"},
    )
    packing: bool = field(
        default=True,
        metadata={"help": "Whether to pack the data into constant length batches to accelerate training"},
    )
    fp16: bool = True
    bf16: bool = False
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    optim: str = "adamw_apex_fused"
    gradient_accumulation_steps: int = 1
    logging_strategy: str = "steps"
    logging_steps: int = 100
    logging_dir: str = "/app/logging_finetuning/",
    save_strategy: str = "no"
    save_steps: int = 1000



@dataclass
class AdapterArguments:
    # TODO: How to get the optimal paratmeters for lora?
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use lora"},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda:["q_proj", "v_proj"],
        metadata={"help": "Target modules for lora"},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "R for lora"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Alpha for lora"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout for lora"},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias for lora"},
    )