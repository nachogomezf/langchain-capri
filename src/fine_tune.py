from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
import torch
import os

def setup_cuda():
    """Setup CUDA device and return appropriate device mapping"""
    if torch.cuda.is_available():
        # Use first available GPU
        device = torch.device("cuda:0")
        device_map = "auto"
        torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    else:
        device = torch.device("cpu")
        device_map = None
    return device, device_map

def load_model():
    device, device_map = setup_cuda()
    model_name = "meta-llama/Llama-3-7b"
    
    print(f"Loading model on device: {device}")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        load_in_8bit=True if device.type == "cuda" else False,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
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

def format_training_example(example):
    """Format the training data to include detailed questionnaire-style information"""
    # Extract structured information from the input if available
    if "\n" in example["input"]:
        # It's already in the detailed format
        formatted_input = example["input"]
    else:
        # Convert simple format to detailed format
        business_info = example["input"].split()
        investment = next((word for word in business_info if word.startswith("$")), "$500K")
        business_type = " ".join(word for word in business_info if not word.startswith("$") and word != "startup")
        
        formatted_input = f"""Generate a business plan for a {investment} {business_type} E-2 Visa startup with the following details:
        
Location: To be determined
Target Market: General market
Competitors: To be analyzed
Unique Value Proposition: Quality service
Initial Team Size: 5
Operating Hours: Standard business hours
Products/Services: Standard industry offerings
Initial Setup Costs: {investment}
Monthly Operating Expenses: To be calculated
Projected First Year Revenue: To be projected"""

    return {
        "input": formatted_input,
        "output": example["output"]
    }

def load_training_data():
    with open("../data/training_data.json", "r") as f:
        training_data = json.load(f)
    # Format each example to match the questionnaire structure
    formatted_data = [format_training_example(example) for example in training_data]
    return Dataset.from_list(formatted_data)

def main():
    # Set PyTorch to use deterministic algorithms
    torch.use_deterministic_algorithms(True)
    
    # Initialize model and tokenizer
    model, tokenizer = load_model()
    model = apply_lora(model)
    dataset = load_training_data()

    # Configure training arguments based on available hardware
    device, _ = setup_cuda()
    batch_size = 4 if device.type == "cuda" else 1
    
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,
        logging_steps=10,
        output_dir="../model/fine_tuned_model",
        save_strategy="epoch",
        learning_rate=2e-4,
        warmup_ratio=0.03,
        gradient_accumulation_steps=4 if device.type == "cuda" else 8,
        fp16=device.type == "cuda",  # Use mixed precision only on GPU
        optim="paged_adamw_8bit" if device.type == "cuda" else "adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    # Start training
    print("Starting training...")
    trainer.train()

    # Save the model and tokenizer
    output_dir = "../model/fine_tuned_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    main()