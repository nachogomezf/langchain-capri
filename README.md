# langchain_business_plan

## Project Description
This repository contains a LangChain-based AI agent that generates **custom business plans** for USA **immigrant investor employment-based visas** (e.g., EB-5, E-2, L-1). The model is **fine-tuned for free** using LoRA on LLaMA 3 and integrates **retrieval-augmented generation (RAG)** for dynamic document adaptation.

## Features
**Customizable Business Plan Generation** - Generates detailed business plans based on visa type and investment details.  
**Fine-Tuned Model (LoRA + LLaMA 3)** - Free fine-tuning on **Google Colab** for improved accuracy.  
**Integration with Existing Business Plan Templates** - Extracts structure from `.docx` templates.  
**Retrieval-Augmented Generation (RAG)** - Pulls real-world examples from previous business plans.  
**Deployable API & UI** - Deploy via **Hugging Face Spaces** or run locally.  

---

## Setup Instructions

### 1️⃣ Install Required Dependencies
```bash
pip install torch transformers peft bitsandbytes datasets accelerate langchain docx2txt streamlit
```

### 2️⃣ Fine-Tune the Model on Google Colab (Free GPU)
#### Load Pretrained LLaMA 3 Model
```python
from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = "meta-llama/Llama-3-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
```
#### Apply LoRA Fine-Tuning
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)
```
#### Prepare Training Data
```python
from datasets import Dataset

training_data = [
    {"input": "Generate an EB-5 business plan for a $1M tech startup.",
     "output": "TechFuture Inc. is a technology company specializing in AI solutions..."},
    {"input": "Create an E-2 visa business plan for a small restaurant.",
     "output": "Gourmet Delights LLC is a fine dining restaurant located in Miami..."}
]

dataset = Dataset.from_list(training_data)
```
#### Train and Save Model
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    output_dir="./fine_tuned_model",
    save_strategy="epoch"
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
```

---

## 3️⃣ Running the Model Locally
```python
from transformers import pipeline

model_path = "fine_tuned_model"
business_plan_generator = pipeline("text-generation", model=model_path)

response = business_plan_generator("Generate a business plan for a $500K E-2 Visa startup", max_length=500)
print(response)
```

---

## 4️⃣ Deploying to Hugging Face (Free API Hosting)
```bash
huggingface-cli login
huggingface-cli upload fine_tuned_model your_model_name
```

---

## 5️⃣ Optional: Web UI with Streamlit
Create a simple UI for users to **upload templates, enter details, and generate business plans**.

### Install Streamlit
```bash
pip install streamlit
```

### Create `app.py`
```python
import streamlit as st
from transformers import pipeline

st.title("📝 AI Business Plan Generator")
st.write("Generate investor visa business plans instantly.")

investment = st.text_input("Investment Amount ($)")
visa_type = st.selectbox("Visa Type", ["EB-5", "E-2", "L-1"])

generate = st.button("Generate Business Plan")

if generate:
    model_path = "fine_tuned_model"
    business_plan_generator = pipeline("text-generation", model=model_path)
    response = business_plan_generator(f"Generate a {visa_type} business plan for a ${investment} startup", max_length=500)
    st.write(response[0]['generated_text'])
```

### Run the Web App
```bash
streamlit run app.py
```

---

## 🚀 Next Steps
1️⃣ **Train with More Business Plans** - Improve quality with additional training data.  
2️⃣ **Add Template Adaptation** - Use `.docx` templates for structured formatting.  
3️⃣ **Enhance RAG Integration** - Retrieve real visa requirements dynamically.  

---

📢 **Need more features? Let me know!** 🚀
