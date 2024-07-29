from typing import Dict, List, Union

class CustomPrompt:
    RESPONSE_TEMPLATE = None

    def __init__(self, user_prompt_format) -> None:
        self.user_prompt_format = user_prompt_format
    
    def format_train_prompt(self, **kwargs):
        return self.user_prompt_format.format(**kwargs)
    
    def format_eval_prompt(self, **kwargs):
        return self.user_prompt_format.format(**kwargs)


class Llama2Prompt:
    TRAIN_TEMPLATE = """<s>[INST] <<SYS>>
{system}
<</SYS>>

{instruction} [/INST] {output}</s>
"""
    DEFAULT_SYSTEM_MSG = "You are a respectful assistant. Your answers will always be concise."
    RESPONSE_TEMPLATE = "[/INST]"

    EVAL_TEMPLATE = f"{TRAIN_TEMPLATE[:TRAIN_TEMPLATE.index(RESPONSE_TEMPLATE)]}{RESPONSE_TEMPLATE}"

    @classmethod
    def format_train_prompt(cls, **kwargs):
        system = cls.DEFAULT_SYSTEM_MSG
        if "system" in kwargs and kwargs["system"] is not None:
            system = kwargs["system"]

        return cls.TRAIN_TEMPLATE.format(
            system=system,
            instruction=kwargs["instruction"],
            output=kwargs["output"])

    @classmethod
    def format_eval_prompt(cls, **kwargs):
        system = cls.DEFAULT_SYSTEM_MSG
        if "system" in kwargs and kwargs["system"] is not None:
            system = kwargs["system"]

        return cls.EVAL_TEMPLATE.format(
            system=system,
            instruction=kwargs["instruction"])


LLM_TEMPLATE = {
    "llama2": Llama2Prompt
}

class PromptFormatter:
    def __init__(self, user_prompt_format):
        if user_prompt_format in LLM_TEMPLATE:
            self.formatter = LLM_TEMPLATE[user_prompt_format]
        else:
            self.formatter = CustomPrompt(user_prompt_format)
        self.response_template = self.formatter.RESPONSE_TEMPLATE

    # this function is used by dataset.map with batched=True option
    def formatting_train_prompts_func(self, example: Dict[str, List[str]]) -> List[str]:
        output_texts = []

        features = list(example)
        for value in zip(*example.values()):
            output_texts.append(self.formatter.format_train_prompt(**dict(zip(features, value))))

        return output_texts

    # this function is used by Evaluator
    def formatting_eval_prompt(self, query: Union[str, Dict[str, str]]) -> str:
        if isinstance(query, str):
            example = {"instruction": query}
        else:
            example = query
        return self.formatter.format_eval_prompt(**example)