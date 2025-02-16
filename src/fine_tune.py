from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

def load_model():
    model_name = "meta-llama/Llama-3-7b"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    return model, tokenizer

def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"]
    )
    return get_peft_model(model, lora_config)

def load_training_data():
    with open("../data/training_data.json", "r") as f:
        training_data = json.load(f)
    return Dataset.from_list(training_data)

def main():
    model, tokenizer = load_model()
    model = apply_lora(model)
    dataset = load_training_data()

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        output_dir="../model/fine_tuned_model",
        save_strategy="epoch"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()

    model.save_pretrained("../model/fine_tuned_model")
    tokenizer.save_pretrained("../model/fine_tuned_model")

if __name__ == "__main__":
    main()