#!/usr/bin/env python3

import os
from huggingface_hub import create_repo, SpaceHardware, SpaceStage, SpaceRuntime

def create_huggingface_space():
    # Define the space name - you can change this to your preferred name
    space_name = "paul-graham-essays-rag"
    
    # Create a new Space configured for Docker
    repo_url = create_repo(
        repo_id=f"cantemizyurek/{space_name}",
        repo_type="space",
        space_runtime=SpaceRuntime.DOCKER,
        space_hardware=SpaceHardware.CPU_BASIC,
        private=False,
        space_stage=SpaceStage.LIVE
    )
    
    print(f"Space created successfully!")
    print(f"Repository URL: {repo_url}")
    print(f"You can clone it with: git clone {repo_url}")
    print(f"Web URL: https://huggingface.co/spaces/cantemizyurek/{space_name}")
    
    return repo_url

if __name__ == "__main__":
    create_huggingface_space()

