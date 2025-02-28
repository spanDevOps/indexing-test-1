#!/usr/bin/env python3
import os
import sys
import gdown
from pathlib import Path

def download_models():
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs
    models = {
        "webface_r50_pfc.onnx": "https://drive.google.com/uc?id=1FPldzmZ6jHfaC-R-jLkxvQRP-cLgxjCT",  # ResNet50 MS1MV3
        "face_occlusion.onnx": "https://drive.google.com/uc?id=1WOrOK-qZO5FcagscCI3td6nnABUPPepD"  # Face occlusion model
    }
    
    for model_name, url in models.items():
        output_path = models_dir / model_name
        if not output_path.exists():
            print(f"Downloading {model_name}...")
            try:
                gdown.download(url, str(output_path), quiet=False)
                if output_path.exists():
                    print(f"Successfully downloaded {model_name}")
                else:
                    print(f"Failed to download {model_name}")
                    sys.exit(1)
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
                sys.exit(1)
        else:
            print(f"{model_name} already exists, skipping download")

if __name__ == "__main__":
    download_models()
