import os

from datetime import datetime
from transformers import TrainerCallback


class SavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

    def on_train_end(self, args, state, control, **kwargs):
        model_path = os.path.join(args.output_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        kwargs["model"].save_pretrained(model_path)


class SavingCallbackHalf(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].half().save_pretrained(checkpoint_path, safe_serialization=True)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

    def on_train_end(self, args, state, control, **kwargs):
        model_path = os.path.join(args.output_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        kwargs["model"].half().save_pretrained(model_path, safe_serialization=True)