#!/usr/bin/env python3

from huggingface_hub import add_space_secret

def add_secrets_to_space():
    """Add required secrets to the Hugging Face Space."""
    space_id = "cantemizyurek/paul-graham-essays-rag"
    
    # Define the secrets to add
    secrets = {
        "HF_LLM_ENDPOINT": "https://nkavqtrvj5mcwad8.us-east-1.aws.endpoints.huggingface.cloud",
        "HF_EMBED_ENDPOINT": "https://w5s43nqb7oa6x3w0.us-east-1.aws.endpoints.huggingface.cloud",
        "HF_TOKEN": "YOUR_HF_TOKEN_HERE"
    }
    
    # Add each secret to the space
    for secret_name, secret_value in secrets.items():
        try:
            add_space_secret(
                repo_id=space_id,
                key=secret_name,
                value=secret_value
            )
            print(f"Successfully added secret: {secret_name}")
        except Exception as e:
            print(f"Error adding secret {secret_name}: {str(e)}")

if __name__ == "__main__":
    add_secrets_to_space()

