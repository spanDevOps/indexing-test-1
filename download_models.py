#!/usr/bin/env python3
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_file(url, target_path):
    """Download a file from URL to target path with progress indicator."""
    print(f"Downloading {url} to {target_path}")
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, target_path, progress)
    print("\nDownload complete!")

def setup_models():
    """Download and set up all required models."""
    # Create necessary directories
    models_dir = Path("/app/src/models")
    insightface_dir = Path("/root/.insightface/models/buffalo_l")
    models_dir.mkdir(parents=True, exist_ok=True)
    insightface_dir.mkdir(parents=True, exist_ok=True)

    # Download InsightFace buffalo_l model
    buffalo_zip = insightface_dir / "buffalo_l.zip"
    if not (insightface_dir / "1k-emb512").exists():
        download_file(
            "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
            buffalo_zip
        )
        # Extract zip
        with zipfile.ZipFile(buffalo_zip, 'r') as zip_ref:
            zip_ref.extractall(insightface_dir)
        # Clean up zip file
        buffalo_zip.unlink()

    # Download face occlusion model
    face_occlusion = models_dir / "face_occlusion.onnx"
    if not face_occlusion.exists():
        download_file(
            "https://huggingface.co/spaces/veai/face-occlusion/resolve/main/face_occlusion.onnx",
            face_occlusion
        )

    # Download WebFace model
    webface = models_dir / "webface_r50_pfc.onnx"
    if not webface.exists():
        download_file(
            "https://huggingface.co/spaces/veai/face-recognition/resolve/main/webface_r50_pfc.onnx",
            webface
        )

if __name__ == "__main__":
    setup_models()
