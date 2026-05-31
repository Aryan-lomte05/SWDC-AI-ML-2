"""
Master Dataset Orchestrator — Extract-and-Combine
===================================================
Spec (bible §5): Manages a 15GB proportional multi-dataset extraction pipeline.
Deletes raw videos immediately after extracting 256x256 face crops to save space.

Supported Datasets:
1. DFDC (Automated via Kaggle CLI)
2. FaceForensics++ (Manual placement required)
3. Celeb-DF v2 (Manual placement required)
4. FakeAVCeleb (Manual placement required)
5. WildDeepfake (Manual placement required)
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import cv2

sys.path.append(str(Path(__file__).parent.parent))
from ml.preprocessing.frame_extractor import FrameExtractor
from ml.preprocessing.face_processor import FaceProcessor

# ── Config ──
MAX_STORAGE_GB = 15.0
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Sub-directories for manual dataset drops
MANUAL_DIRS = ["ff++", "celeb-df", "fakeavceleb", "wilddeepfake", "kodf"]

def setup_directories():
    """Ensure all expected input and output directories exist."""
    PROCESSED_DIR.joinpath("real").mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.joinpath("fake").mkdir(parents=True, exist_ok=True)
    for d in MANUAL_DIRS:
        RAW_DIR.joinpath(d).mkdir(parents=True, exist_ok=True)

def get_dir_size_gb(directory: Path) -> float:
    if not directory.exists(): return 0.0
    return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file()) / (1024 ** 3)

# ── Dataset Processing Pipeline ──

def process_video_directory(source_dir: Path, dataset_name: str, label_map: dict = None):
    """
    Extracts face crops from all .mp4 files in a directory, saves them, 
    and IMMEDIATELY DELETES the raw video to save gigabytes of space.
    """
    videos = list(source_dir.glob("*.mp4"))
    if not videos:
        return

    print(f"\n⚙️ Extracting faces from {dataset_name} ({len(videos)} videos)...")
    extractor = FrameExtractor(target_fps=2, max_frames=10) # 10 frames per video max
    processor = FaceProcessor(output_size=256)
    
    real_out = PROCESSED_DIR / "real"
    fake_out = PROCESSED_DIR / "fake"
    
    for vid_path in tqdm(videos, desc=f"{dataset_name}"):
        if get_dir_size_gb(DATA_DIR) >= MAX_STORAGE_GB:
            print(f"\n⚠️ Reached global {MAX_STORAGE_GB}GB cap! Halting extraction.")
            return

        # Determine label (Default to FAKE if no label_map provided, or check filename)
        is_fake = True
        if label_map and vid_path.name in label_map:
            is_fake = (label_map[vid_path.name].upper() == "FAKE")
        elif "real" in vid_path.name.lower() or "original" in str(vid_path).lower():
            is_fake = False
            
        out_dir = fake_out if is_fake else real_out
        
        try:
            frames = extractor.extract_frames(str(vid_path))
            for i, frame in enumerate(frames):
                faces = processor.process_frame(frame)
                if faces and faces[0]["aligned_crop"] is not None:
                    save_path = out_dir / f"{dataset_name}_{vid_path.stem}_f{i:03d}.jpg"
                    cv2.imwrite(str(save_path), faces[0]["aligned_crop"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception:
            pass
            
        # 🗑️ CRITICAL: Delete the raw video to prevent disk bloat
        try:
            os.remove(vid_path)
        except OSError:
            pass

# ── DFDC Module (Automated) ──

def handle_dfdc():
    dfdc_dir = RAW_DIR / "dfdc"
    dfdc_dir.mkdir(exist_ok=True)
    zip_path = dfdc_dir / "train_sample_videos.zip"
    
    if not list(dfdc_dir.glob("*.mp4")):
        print("\n📥 Downloading DFDC sample (~4GB) via Kaggle...")
        try:
            subprocess.run(["kaggle", "competitions", "download", "-c", "deepfake-detection-challenge", "-f", "train_sample_videos.zip", "-p", str(dfdc_dir)], check=True)
            print("📦 Unzipping DFDC...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract directly into dfdc_dir
                for member in zip_ref.namelist():
                    if member.endswith('.mp4') or member.endswith('.json'):
                        filename = os.path.basename(member)
                        if not filename: continue
                        with zip_ref.open(member) as source, open(dfdc_dir / filename, "wb") as target:
                            target.write(source.read())
            os.remove(zip_path)
        except Exception as e:
            print(f"❌ DFDC download skipped/failed: {e}")
            return

    # Process
    label_map = {}
    meta_path = dfdc_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            data = json.load(f)
            label_map = {k: v.get("label", "FAKE") for k, v in data.items()}
            
    process_video_directory(dfdc_dir, "DFDC", label_map)

# ── Main ──

def main():
    print("========================================================")
    print("  DeepScan Dataset Orchestrator (Extract-and-Combine)   ")
    print("========================================================")
    
    setup_directories()
    
    # 1. Run Automated DFDC
    handle_dfdc()
    
    # 2. Process any manual drops
    for d in MANUAL_DIRS:
        process_video_directory(RAW_DIR / d, d.upper())
        
    print("\n========================================================")
    print(f"✅ Pipeline complete. Current Dataset Size: {get_dir_size_gb(DATA_DIR):.2f} GB")
    print("\nℹ️ HOW TO ADD MORE DATASETS (FaceForensics++, Celeb-DF, etc.):")
    print(f"   Download raw .mp4 videos and drop them into:")
    for d in MANUAL_DIRS:
        print(f"   - {RAW_DIR / d}")
    print("   Then run this script again. It will extract the faces and instantly delete the massive videos.")

if __name__ == "__main__":
    main()
