"""
DFDC (Deepfake Detection Challenge) - Dataset Preparation Pipeline
==================================================================
Spec (bible §5): Downloads the DFDC sample dataset, caps extraction to < 10GB,
and prepares the data for training.

Requires: Kaggle API key installed in ~/.kaggle/kaggle.json
"""

import os
import json
import zipfile
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import sys

# Ensure root is in pythonpath
sys.path.append(str(Path(__file__).parent.parent))
from ml.preprocessing.frame_extractor import FrameExtractor
from ml.preprocessing.face_processor import FaceProcessor
import cv2

MAX_STORAGE_GB = 10.0
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "dfdc"
PROCESSED_DIR = DATA_DIR / "processed"

def check_kaggle_auth():
    """Verify kaggle credentials exist."""
    kaggle_dir = Path.home() / ".kaggle"
    if not (kaggle_dir / "kaggle.json").exists():
        print("❌ Kaggle API key not found!")
        print("To download the DFDC dataset from Kaggle:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New Token' to download kaggle.json")
        print(f"3. Place kaggle.json in: {kaggle_dir}")
        print("4. Re-run this script.")
        sys.exit(1)

def get_dir_size_gb(directory: Path) -> float:
    """Calculate directory size in GB."""
    if not directory.exists(): return 0.0
    total = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
    return total / (1024 ** 3)

def download_dfdc_sample():
    """Download the 4GB DFDC sample dataset via Kaggle CLI."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "train_sample_videos.zip"
    
    if zip_path.exists() or (RAW_DIR / "train_sample_videos").exists():
        print("✅ DFDC raw dataset already present.")
        return
        
    print("📥 Downloading DFDC sample dataset (~4GB) via Kaggle...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "kaggle"
        ], check=True, stdout=subprocess.DEVNULL)
        
        subprocess.run([
            "kaggle", "competitions", "download", 
            "-c", "deepfake-detection-challenge", 
            "-f", "train_sample_videos.zip",
            "-p", str(RAW_DIR)
        ], check=True)
        
        print("📦 Unzipping dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
            
        # Delete zip to save space
        os.remove(zip_path)
    except Exception as e:
        print(f"❌ Failed to download/extract dataset: {e}")
        sys.exit(1)

def extract_and_align_faces():
    """Extract frames and align faces, stopping before we hit 10GB."""
    real_dir = PROCESSED_DIR / "real"
    fake_dir = PROCESSED_DIR / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    video_dir = RAW_DIR / "train_sample_videos"
    if not video_dir.exists():
        print("❌ Extracted videos not found.")
        return
        
    # Read metadata.json to get labels
    meta_path = video_dir / "metadata.json"
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    videos = list(video_dir.glob("*.mp4"))
    
    extractor = FrameExtractor(target_fps=5, max_frames=20)
    processor = FaceProcessor(output_size=256)
    
    print(f"⚙️ Starting face extraction (Target Cap: {MAX_STORAGE_GB} GB)")
    
    processed_count = 0
    for vid_path in tqdm(videos, desc="Processing videos"):
        # Check storage constraints every 10 videos
        if processed_count % 10 == 0:
            current_size = get_dir_size_gb(DATA_DIR)
            if current_size >= MAX_STORAGE_GB * 0.95:  # 5% buffer
                print(f"\n⚠️ Reached storage cap of {MAX_STORAGE_GB}GB. Stopping extraction.")
                break
                
        label = metadata.get(vid_path.name, {}).get("label", "FAKE")
        out_dir = fake_dir if label == "FAKE" else real_dir
        
        try:
            frames = extractor.extract_frames(str(vid_path))
            for i, frame in enumerate(frames):
                faces = processor.process_frame(frame)
                if faces and faces[0]["aligned_crop"] is not None:
                    # Save the first detected face
                    crop = faces[0]["aligned_crop"]
                    save_path = out_dir / f"{vid_path.stem}_frame{i:03d}.jpg"
                    cv2.imwrite(str(save_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            continue
            
        processed_count += 1
        
    print(f"✅ Data processing complete. Final dataset size: {get_dir_size_gb(DATA_DIR):.2f} GB")

if __name__ == "__main__":
    check_kaggle_auth()
    download_dfdc_sample()
    extract_and_align_faces()
