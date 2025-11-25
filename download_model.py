#!/usr/bin/env python3
"""Download Kokoro-82M model files from HuggingFace."""

import os
from huggingface_hub import hf_hub_download

# Create directories
os.makedirs('models/kokoro-82m/voices', exist_ok=True)

print("Downloading Kokoro-82M model files...")

# Download files
hf_hub_download('hexgrad/Kokoro-82M', 'config.json', local_dir='models/kokoro-82m')
print("✓ config.json")

hf_hub_download('hexgrad/Kokoro-82M', 'kokoro-v1_0.pth', local_dir='models/kokoro-82m')
print("✓ kokoro-v1_0.pth")

hf_hub_download('hexgrad/Kokoro-82M', 'voices/af_heart.pt', local_dir='models/kokoro-82m')
print("✓ voices/af_heart.pt")

print("\n✅ All model files downloaded successfully!")
