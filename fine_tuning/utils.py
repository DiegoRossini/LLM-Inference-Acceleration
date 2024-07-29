from typing import List
from peft import PeftModel

def merge_peft_model(model, adapter_path: List[str]):
    for path in adapter_path:
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    return model