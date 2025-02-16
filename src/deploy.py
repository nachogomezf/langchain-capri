from huggingface_hub import login, upload_folder
import os
import argparse

def deploy_model(model_path, model_name):
    login()  # Will use the token from huggingface-cli login
    upload_folder(
        folder_path=model_path,
        repo_id=model_name,
        repo_type="model"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Name for the model on HuggingFace Hub")
    args = parser.parse_args()
    
    deploy_model("../model/fine_tuned_model", args.model_name)