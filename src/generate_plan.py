from transformers import pipeline
from huggingface_hub import InferenceApi
import os

def generate_business_plan(prompt, max_length=2000):
    """Generate business plan using Hugging Face's Inference API"""
    api_token = os.getenv("HUGGINGFACE_TOKEN")
    model_id = os.getenv("HUGGINGFACE_MODEL_ID", "your-model-id")  # Replace with your model ID after uploading
    
    # Initialize the inference API client
    inference = InferenceApi(
        repo_id=model_id,
        token=api_token,
        task="text-generation"
    )
    
    # Generate using the hosted model
    response = inference(
        prompt,
        max_length=max_length,
        temperature=0.7,
        num_return_sequences=1
    )
    
    return response[0]['generated_text']

if __name__ == "__main__":
    prompt = "Generate a business plan for a $500K E-2 Visa startup"
    response = generate_business_plan(prompt)
    print(response)