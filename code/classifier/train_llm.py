import os
import hydra
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer
from data_loader import create_prompt
from datasets import Dataset


class InstructTuneTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lora_cfg = cfg["lora"]
        self.quant_cfg = cfg["quant"]
        self.model_cfg = cfg["model"]
        self.train_cfg = cfg["train"]
        self.eval_cfg = cfg["eval"]
        self.peft_config = None
        self.tokenizer = None
        self.model = None

    def load_dataset(self):
        train_dataset = Dataset.from_json(self.train_cfg["train_data_file"])
        val_dataset = Dataset.from_json(self.train_cfg["val_data_file"])
        print("train dataset:")
        print(train_dataset)
        return train_dataset, val_dataset

    def load_model_and_tokenizer_for_train(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=self.quant_cfg["use_4bit"],
            bnb_4bit_quant_type=self.quant_cfg["quant_type"],
            bnb_4bit_use_double_quant=self.quant_cfg["use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16 if self.quant_cfg["use_bf16"] else None
        )

        self.peft_config = LoraConfig(
            lora_alpha=self.lora_cfg["alpha"],
            lora_dropout=self.lora_cfg["dropout"],
            r=self.lora_cfg["rank"],
            bias=self.lora_cfg["bias"],
            target_modules=self.lora_cfg["target_modules"],
            task_type=self.lora_cfg["task_type"]
        )

        device_map = {"": Accelerator().local_process_index}

        model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg["model_name"],
            device_map=device_map,
            quantization_config=nf4_config,
            use_cache=False,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if self.quant_cfg["use_bf16"] else None,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg["model_name"],
            padding_side="right",
            truncation_side="right"
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.peft_config)
        model.print_trainable_parameters()
        model.config.torch_dtype = torch.float32

        self.model = model
        self.tokenizer = tokenizer

    def train(self, train_dataset: Dataset, val_dataset: Dataset):

        args = TrainingArguments(
            output_dir=self.model_cfg["model_output_dir"],
            num_train_epochs=self.train_cfg["num_epochs"],
            per_device_train_batch_size=self.train_cfg["batch_size"],
            eval_steps=self.train_cfg["eval_steps"],
            learning_rate=self.train_cfg["learning_rate"],
            bf16=self.quant_cfg["use_bf16"],
            lr_scheduler_type=self.train_cfg["lr_scheduler_type"],
            warmup_ratio=self.train_cfg["warmup_ratio"],
            resume_from_checkpoint=None
        )

        trainer = SFTTrainer(
            model=self.model,
            peft_config=self.peft_config,
            max_seq_length=self.model_cfg["max_length"],
            tokenizer=self.tokenizer,
            packing=True,
            formatting_func=create_prompt,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        trainer.accelerator.print(f"{trainer.model}")

        trainer.train()
        trainer.save_model()

    def evaluate(self, val_dataset: Dataset):
        pass


def generate(prompt, tokenizer, model, max_new_tokens=10):
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_output = tokenizer.batch_decode(generated_ids)
    return decoded_output[0].replace(prompt, "")


@hydra.main(version_base=None, config_path="../../config", config_name="conf_llm")
def main(cfg):
    cfg.ddp_find_unused_parameters = False
    cfg.local_rank = -1

    trainer = InstructTuneTrainer(cfg)
    train_dataset, val_dataset = trainer.load_dataset()
    trainer.load_model_and_tokenizer_for_train()
    trainer.train(train_dataset, val_dataset)

    """
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])
    prompt = ""
    generate(prompt, tokenizer, model)
    """


if __name__ == "__main__":
    main()