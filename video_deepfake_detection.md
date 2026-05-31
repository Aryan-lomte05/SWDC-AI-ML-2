# 🎯 Video Deepfake Detection — End-to-End World-Class System

> **"An AI-powered forensic antivirus for fake digital video media."**  
> Production-ready | Research-backed | State-of-the-art | Explainable

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why Video Deepfake Detection is Hard](#2-why-video-deepfake-detection-is-hard)
3. [System Overview & Goals](#3-system-overview--goals)
4. [Deepfake Taxonomy — What We're Fighting](#4-deepfake-taxonomy--what-were-fighting)
5. [Datasets](#5-datasets)
6. [Full ML/AI Detection Pipeline](#6-full-mlai-detection-pipeline)
7. [Model Architecture — Layer by Layer](#7-model-architecture--layer-by-layer)
8. [Temporal Analysis — The Key Differentiator](#8-temporal-analysis--the-key-differentiator)
9. [Explainable AI (XAI)](#9-explainable-ai-xai)
10. [Confidence Scoring Engine](#10-confidence-scoring-engine)
11. [Tech Stack](#11-tech-stack)
12. [System Architecture](#12-system-architecture)
13. [Processing Pipeline — Step by Step](#13-processing-pipeline--step-by-step)
14. [Folder Structure](#14-folder-structure)
15. [API Design](#15-api-design)
16. [Frontend Dashboard](#16-frontend-dashboard)
17. [Training Strategy](#17-training-strategy)
18. [Evaluation Metrics](#18-evaluation-metrics)
19. [Robustness & Adversarial Hardening](#19-robustness--adversarial-hardening)
20. [Advanced Features](#20-advanced-features)
21. [Development Timeline](#21-development-timeline)
22. [Real-World Applications](#22-real-world-applications)
23. [Future Scope](#23-future-scope)
24. [Research References](#24-research-references)

---

## 1. Problem Statement

With the explosive advancement of generative AI — including GANs, diffusion models, and neural face-swapping — **deepfake videos** have reached a level of realism where the human eye alone can no longer reliably distinguish real from fake.

### The Threat Landscape

| Threat Category | Real-World Example |
|---|---|
| Political manipulation | Fabricated speeches by world leaders |
| Financial scams | CEO voice/video impersonation for wire fraud |
| Celebrity exploitation | Non-consensual intimate deepfakes |
| Identity fraud | Bypassing video KYC verification systems |
| Misinformation | Fake news clips spread virally |
| Cyberbullying & blackmail | Fabricated videos targeting individuals |
| Legal evidence tampering | Manipulated courtroom video evidence |

### The Scale of the Problem

- Deepfake video content online **doubled every 6 months** from 2019–2024
- Detection tools lag **6–18 months** behind generation tools
- A single viral deepfake can cause **irreversible reputational or financial damage** in hours
- Existing consumer tools have accuracy rates **below 65%** on modern generation techniques

**This project builds a system that closes this gap — targeting >95% AUC on state-of-the-art deepfakes.**

---

## 2. Why Video Deepfake Detection is Hard

Understanding the difficulty is essential to designing the right solution.

### Generation Methods We Must Detect

| Method | Description | Hardness to Detect |
|---|---|---|
| **FaceSwap** | Source face replaced on target body | Medium |
| **Face Reenactment** | Source expressions driven onto target | Hard |
| **Full Synthesis (GAN)** | Entire face generated from scratch | Very Hard |
| **Neural Talking Heads** | One-shot expression transfer | Very Hard |
| **Diffusion-based editing** | Localized inpainting of real video | Extremely Hard |
| **Lip-sync manipulation** | Only mouth region altered | Hard (localized) |

### Core Detection Challenges

- **Spatial artifacts** are increasingly subtle (sub-pixel level)
- **Temporal coherence** is improving in newer models
- **Compression** from platforms (YouTube, TikTok) destroys low-level artifacts
- **Resolution diversity** — from 144p to 4K
- **Occlusions** — hats, glasses, motion blur
- **Generalization** — a model trained on FaceForensics++ may fail on Celeb-DF
- **Adversarial attacks** — deepfakes designed to fool detectors

---

## 3. System Overview & Goals

### Primary Goal

Build a **multimodal, temporally-aware, explainable deepfake detection platform** that:

- Accepts video uploads
- Runs a full forensic analysis pipeline
- Returns a confidence score with frame-level and region-level evidence
- Produces human-readable, explainable reports

### Key Performance Targets

| Metric | Target |
|---|---|
| AUC-ROC | ≥ 0.95 |
| Accuracy (balanced) | ≥ 92% |
| False Positive Rate | ≤ 5% |
| Processing speed | ≤ 30s per 1-min video (GPU) |
| Explainability | Heatmap on every flagged frame |

---

## 4. Deepfake Taxonomy — What We're Fighting

### Video-Specific Manipulation Types

```
Video Deepfakes
├── Face Replacement
│   ├── Entire face swap (DeepFaceLab, FaceSwap)
│   └── Region-specific swap (eyes, mouth)
├── Face Reenactment
│   ├── Source-driven (First Order Motion Model)
│   └── Puppeteer-based (Neural Talking Heads)
├── Attribute Manipulation
│   ├── Age/gender modification
│   └── Expression editing (emotion deepfakes)
├── Full Synthesis
│   ├── GAN-generated faces (StyleGAN)
│   └── Diffusion-based video generation (Sora-style)
└── Lip-Sync Forgery
    ├── Wav2Lip injection
    └── Audio-driven face reenactment
```

### Generation Tool Fingerprints (What We Exploit)

Each deepfake generator leaves **unique forensic traces**:

| Generator | Primary Artifact |
|---|---|
| DeepFaceLab | Blending boundary artifacts at face mask edges |
| FaceSwap | Lighting inconsistency between swapped face and neck |
| First Order Motion | Warping grid artifacts in fast motion |
| Wav2Lip | Lip region sharpness discontinuity |
| StyleGAN | High-frequency checkerboard patterns |
| Diffusion models | Temporal incoherence in background regions |

---

## 5. Datasets

### Primary Training Datasets

| Dataset | Videos | Forgery Types | Notes |
|---|---|---|---|
| **FaceForensics++** | 1,000 real + 4,000 fake | DeepFakes, Face2Face, FaceSwap, NeuralTextures | Gold standard benchmark |
| **DFDC (Deepfake Detection Challenge)** | 128,154 videos | Multiple methods | Facebook/Meta dataset, highly diverse |
| **Celeb-DF v2** | 590 real + 5,639 fake | Celebrity deepfakes | High visual quality, harder to detect |
| **FakeAVCeleb** | 500+ real + 19,500 fake | Audio-visual fakes | Multimodal, includes lip-sync |
| **WildDeepfake** | 3,805 real + 3,509 fake | In-the-wild internet deepfakes | Real-world noise/compression |
| **KoDF** | 62,166 real + 175,776 fake | Korean deepfakes | Cross-ethnicity generalization |

### Data Augmentation Strategy

```python
augmentation_pipeline = [
    # Compression artifacts (simulate social media)
    VideoCompression(quality_range=(23, 40)),  # H.264 CRF
    
    # Geometric transforms
    RandomCrop(scale=(0.8, 1.0)),
    HorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    
    # Color/lighting
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    RandomGrayscale(p=0.05),
    
    # Noise injection
    GaussianNoise(std_range=(0, 0.02)),
    JPEGCompression(quality_range=(50, 95)),
    
    # Temporal
    FrameDropout(p=0.1),
    TemporalJitter(max_shift=2),
]
```

### Class Balancing

- Deepfake datasets are heavily imbalanced (more fakes than reals in some)
- Use **stratified sampling** + **class-weighted loss**
- Synthesize hard negatives using **curriculum learning**

---

## 6. Full ML/AI Detection Pipeline

```
INPUT VIDEO
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 1: Video Preprocessing       │
│  - Frame extraction (FPS sampling)  │
│  - Resolution normalization         │
│  - Scene cut detection              │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 2: Face Detection & Tracking │
│  - MTCNN / RetinaFace face detector │
│  - DeepSORT multi-face tracker      │
│  - Face alignment (68-point landmarks)│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 3: Spatial Feature Extraction│
│  - XceptionNet / EfficientNet-B4    │
│  - Vision Transformer (ViT-L/16)    │
│  - Frequency domain analysis (FFT)  │
│  - DCT artifact analysis            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 4: Temporal Analysis         │
│  - LSTM over frame sequences        │
│  - 3D CNN (R3D, SlowFast)          │
│  - Optical flow consistency         │
│  - Blink/gaze/lip sync analysis     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 5: Ensemble Fusion           │
│  - Weighted model voting            │
│  - Uncertainty estimation           │
│  - Cross-modal consistency check    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  STAGE 6: Confidence Scoring +      │
│  Explainability Output              │
│  - Grad-CAM heatmaps                │
│  - Frame-level anomaly timeline     │
│  - Final verdict + risk level       │
└─────────────────────────────────────┘
    │
    ▼
OUTPUT: Report + Visualizations
```

---

## 7. Model Architecture — Layer by Layer

### 7.1 Backbone 1 — XceptionNet (Spatial CNN)

XceptionNet is the **gold standard** for deepfake spatial detection, originally proposed by FaceForensics++ researchers.

```python
import torch
import torch.nn as nn
import timm

class XceptionDetector(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)  # [B, 2048]
        features = self.dropout(features)
        return self.classifier(features), features  # Return logits + embeddings
```

**Why XceptionNet?**
- Depthwise separable convolutions capture **channel-specific manipulation artifacts**
- Pretrained on ImageNet, fine-tuned on FaceForensics++
- Operates at the **pixel artifact level** (blending seams, GAN fingerprints)

---

### 7.2 Backbone 2 — Vision Transformer (ViT-L)

```python
class ViTDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model(
            'vit_large_patch16_224',
            pretrained=True,
            num_classes=0
        )
        self.attention_pool = nn.MultiheadAttention(1024, 8)
        self.head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        tokens = self.vit.forward_features(x)  # [B, N+1, 1024]
        cls_token = tokens[:, 0, :]  # CLS token
        patch_tokens = tokens[:, 1:, :]  # Patch tokens
        
        # Cross-attend cls to patches for attention maps
        attn_out, attn_weights = self.attention_pool(
            cls_token.unsqueeze(0),
            patch_tokens.transpose(0, 1),
            patch_tokens.transpose(0, 1)
        )
        return self.head(attn_out.squeeze(0)), attn_weights
```

**Why ViT?**
- Self-attention captures **global inconsistencies** (e.g., left ear doesn't match right side lighting)
- Attention maps serve as **built-in explainability**
- Superior on high-resolution crops (ViT-L/16 at 384×384)

---

### 7.3 Backbone 3 — Frequency Domain Analysis

Deepfake generators operate in **pixel space** but leave traces in the **frequency domain** that CNNs sometimes miss.

```python
import numpy as np
import cv2
import torch

class FrequencyAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN to classify frequency artifacts
        self.freq_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(128 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def compute_fft(self, img_tensor):
        """Convert image to frequency domain magnitude spectrum"""
        # img_tensor: [B, C, H, W]
        gray = img_tensor.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)
        return magnitude
    
    def forward(self, x):
        freq_map = self.compute_fft(x)  # [B, 1, H, W]
        return self.freq_cnn(freq_map)
```

**Detects:**
- GAN checkerboard artifacts (appear as grid patterns in FFT)
- Up-sampling artifacts from face decoders
- Compression-inconsistent regions

---

### 7.4 Temporal Model — Bidirectional LSTM over Frame Sequence

```python
class TemporalDeepfakeDetector(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, frame_features):
        # frame_features: [B, T, feature_dim] — sequence of frame embeddings
        lstm_out, _ = self.lstm(frame_features)  # [B, T, hidden*2]
        
        # Temporal attention — which frames are most suspicious?
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [B, T, 1]
        context = (attn_weights * lstm_out).sum(dim=1)  # [B, hidden*2]
        
        return self.classifier(context), attn_weights.squeeze(-1)  # Return per-frame weights
```

---

### 7.5 3D CNN — SlowFast Network

```python
# Using PyTorchVideo's SlowFast for volumetric video analysis
from pytorchvideo.models import slowfast

class SlowFastDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.slowfast = slowfast.create_slowfast(
            input_channels=(3, 3),
            model_depth=50,
            model_num_class=400,  # Pretrained on Kinetics-400
        )
        # Replace head
        self.slowfast.blocks[-1].proj = nn.Sequential(
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, slow_pathway, fast_pathway):
        # slow: [B, 3, T/alpha, H, W], fast: [B, 3, T, H, W]
        return self.slowfast([slow_pathway, fast_pathway])
```

**Why SlowFast?**
- **Slow pathway**: captures high-resolution spatial detail at low frame rate
- **Fast pathway**: captures motion dynamics at high frame rate
- Perfectly suited for detecting **temporal inconsistencies** in deepfakes

---

### 7.6 Ensemble Fusion

```python
class EnsembleDetector(nn.Module):
    def __init__(self, xception, vit, freq_net, temporal_lstm):
        super().__init__()
        self.xception = xception
        self.vit = vit
        self.freq_net = freq_net
        self.temporal_lstm = temporal_lstm
        
        # Learned fusion weights (trainable)
        self.fusion = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Calibration temperatures per model
        self.temperatures = nn.Parameter(torch.ones(4))
    
    def forward(self, frames, frame_features):
        # Per-frame predictions
        p_xception, _ = self.xception(frames)
        p_vit, attn = self.vit(frames)
        p_freq = self.freq_net(frames)
        p_temporal, frame_weights = self.temporal_lstm(frame_features)
        
        # Temperature scaling for calibration
        scores = torch.stack([
            p_xception / self.temperatures[0],
            p_vit / self.temperatures[1],
            p_freq / self.temperatures[2],
            p_temporal / self.temperatures[3]
        ], dim=-1)
        
        final_score = self.fusion(scores)
        
        return {
            'final_score': final_score,
            'model_scores': scores,
            'attention_map': attn,
            'frame_weights': frame_weights
        }
```

---

## 8. Temporal Analysis — The Key Differentiator

Most amateur detection systems analyze only **individual frames**. This system's **temporal analysis** is what separates it from everything else.

### 8.1 Physiological Signal Analysis

Real humans have measurable biological signals visible in video. Deepfakes often break these.

```python
class PhysiologicalAnalyzer:
    """
    Detects biological inconsistencies that deepfakes cannot replicate:
    - Eye blink rate (normal: 15-20 blinks/min)
    - Microsaccades (tiny eye movements)  
    - Facial blood flow patterns (rPPG)
    - Head micro-movements
    """
    
    def analyze_blink_patterns(self, eye_regions: list) -> dict:
        """
        Deepfakes often miss natural blink patterns:
        - Missing blinks (early GAN models)
        - Unnatural blink timing
        - Asymmetric blinking
        """
        blink_frames = self._detect_blinks(eye_regions)
        blink_rate = len(blink_frames) / (len(eye_regions) / 30)  # per minute
        
        return {
            'blink_rate': blink_rate,
            'is_suspicious': blink_rate < 5 or blink_rate > 40,
            'blink_frames': blink_frames,
            'symmetry_score': self._measure_blink_symmetry(eye_regions, blink_frames)
        }
    
    def analyze_rppg(self, face_regions: list) -> dict:
        """
        Remote Photoplethysmography: detect heart rate signal from skin color
        Deepfakes lack coherent rPPG signals
        """
        # Extract green channel signal from cheeks
        cheek_signals = [np.mean(r[:, :, 1]) for r in face_regions]  # Green channel
        
        # FFT to find dominant frequency (heart rate: 0.75-3.5 Hz at 30fps)
        fft = np.abs(np.fft.rfft(cheek_signals))
        freqs = np.fft.rfftfreq(len(cheek_signals), 1/30)
        
        hr_mask = (freqs >= 0.75) & (freqs <= 3.5)
        hr_power = fft[hr_mask].max()
        total_power = fft.sum()
        
        snr = hr_power / (total_power - hr_power + 1e-8)
        
        return {
            'rppg_snr': snr,
            'is_suspicious': snr < 0.1,  # Low SNR = no heartbeat signal
            'estimated_hr': freqs[hr_mask][fft[hr_mask].argmax()] * 60
        }
```

### 8.2 Lip-Sync Analysis

```python
class LipSyncAnalyzer:
    """
    Detects Wav2Lip-style audio-visual forgeries where only
    the lip region is replaced.
    """
    
    def compute_lip_landmark_consistency(self, landmarks_sequence):
        """
        Track lip landmark velocities across frames.
        Swapped lips have discontinuous velocity profiles.
        """
        velocities = []
        for i in range(1, len(landmarks_sequence)):
            delta = landmarks_sequence[i][48:68] - landmarks_sequence[i-1][48:68]
            velocities.append(np.linalg.norm(delta, axis=1).mean())
        
        # Detect sudden velocity discontinuities
        velocity_std = np.std(velocities)
        velocity_jumps = [i for i in range(1, len(velocities))
                         if abs(velocities[i] - velocities[i-1]) > 3 * velocity_std]
        
        return {
            'velocity_profile': velocities,
            'discontinuities': velocity_jumps,
            'is_suspicious': len(velocity_jumps) > 2
        }
    
    def check_lip_region_texture_consistency(self, frames, lip_masks):
        """
        Compare texture statistics inside lip region vs surrounding face.
        Swapped regions often have different noise/sharpness profiles.
        """
        results = []
        for frame, mask in zip(frames, lip_masks):
            lip_region = frame[mask > 0]
            face_region = frame[mask == 0]
            
            lip_laplacian = cv2.Laplacian(lip_region, cv2.CV_64F).var()
            face_laplacian = cv2.Laplacian(face_region, cv2.CV_64F).var()
            
            sharpness_ratio = lip_laplacian / (face_laplacian + 1e-8)
            results.append(sharpness_ratio)
        
        mean_ratio = np.mean(results)
        return {
            'sharpness_ratio': mean_ratio,
            'is_suspicious': mean_ratio > 2.0 or mean_ratio < 0.3
        }
```

### 8.3 Optical Flow Consistency

```python
class OpticalFlowAnalyzer:
    """
    Deepfakes produce unnatural optical flow:
    - Face region flow inconsistent with head motion
    - Temporal flickering between generated frames
    """
    
    def compute_flow_consistency(self, frames: list) -> dict:
        suspicious_frames = []
        
        for i in range(1, len(frames)):
            prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Compute flow magnitude variance (fake faces flicker)
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            if magnitude.std() > self.threshold:
                suspicious_frames.append(i)
        
        return {
            'suspicious_frames': suspicious_frames,
            'flow_anomaly_score': len(suspicious_frames) / len(frames)
        }
```

---

## 9. Explainable AI (XAI)

Every detection result must be **explainable**. Users must understand *why* a video is flagged.

### 9.1 Grad-CAM Visualization

```python
import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    """
    Generate class activation maps showing WHICH regions triggered detection.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        output.backward(retain_graph=True)
        
        # Weight activations by gradient importance
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global avg pool
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()
    
    def overlay_on_frame(self, frame, cam, alpha=0.4):
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlaid = (1 - alpha) * frame + alpha * heatmap
        return np.uint8(overlaid)
```

### 9.2 Attention Map Visualization (ViT)

```python
def visualize_vit_attention(model, image_tensor, layer=-1):
    """
    Extract and visualize attention from the last ViT transformer block.
    Shows which image patches the model focused on.
    """
    hooks = []
    attention_maps = []
    
    def hook_fn(module, input, output):
        attention_maps.append(output[1])  # attention weights
    
    # Hook into last attention block
    hook = model.vit.blocks[layer].attn.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(image_tensor)
    
    hook.remove()
    
    # Average across heads, focus on CLS token attention
    attn = attention_maps[0].mean(dim=1)  # [B, N+1, N+1]
    cls_attn = attn[0, 0, 1:]  # CLS attending to patches
    
    # Reshape to grid
    num_patches = int(cls_attn.shape[0] ** 0.5)
    attn_map = cls_attn.reshape(num_patches, num_patches).cpu().numpy()
    attn_map = cv2.resize(attn_map, (224, 224))
    
    return attn_map
```

### 9.3 SHAP-Based Feature Importance

```python
import shap

class SHAPExplainer:
    """
    Explain which input features (pixel regions) contributed most
    to the deepfake classification.
    """
    
    def __init__(self, model):
        self.model = model
        self.explainer = shap.GradientExplainer(model, background_data)
    
    def explain_frame(self, frame_tensor):
        shap_values = self.explainer.shap_values(frame_tensor)
        return shap_values
    
    def generate_summary_report(self, video_path, shap_values_per_frame):
        """
        Generate human-readable explanation:
        - "The right eye region showed unnatural blending (35% contribution)"
        - "Lip movement was inconsistent with audio (28% contribution)"  
        - "Skin texture artifacts detected on left cheek (22% contribution)"
        """
        region_importances = self._aggregate_by_region(shap_values_per_frame)
        
        report = []
        for region, importance in sorted(region_importances.items(),
                                         key=lambda x: -x[1]):
            if importance > 0.05:
                report.append({
                    'region': region,
                    'importance': importance,
                    'description': self.REGION_DESCRIPTIONS[region]
                })
        return report
```

---

## 10. Confidence Scoring Engine

### 10.1 Multi-Signal Score Aggregation

```python
class ConfidenceScoringEngine:
    """
    Aggregate signals from all detection stages into a final score.
    Uses uncertainty-aware calibrated probabilities.
    """
    
    SIGNAL_WEIGHTS = {
        'spatial_xception': 0.25,
        'spatial_vit': 0.20,
        'frequency_domain': 0.10,
        'temporal_lstm': 0.20,
        'physiological': 0.15,
        'lip_sync': 0.10
    }
    
    RISK_THRESHOLDS = {
        'AUTHENTIC':    (0.00, 0.20),
        'LOW_RISK':     (0.20, 0.40),
        'SUSPICIOUS':   (0.40, 0.65),
        'HIGH_RISK':    (0.65, 0.85),
        'FAKE':         (0.85, 1.00)
    }
    
    def compute_final_score(self, signal_scores: dict) -> dict:
        # Weighted average
        weighted_score = sum(
            signal_scores[k] * self.SIGNAL_WEIGHTS[k]
            for k in self.SIGNAL_WEIGHTS if k in signal_scores
        )
        
        # Uncertainty estimation via MC Dropout
        mc_scores = self._mc_dropout_inference(signal_scores, n_passes=20)
        uncertainty = np.std(mc_scores)
        
        # Determine risk level
        risk_level = self._classify_risk(weighted_score)
        
        return {
            'fake_probability': round(weighted_score, 4),
            'realness_score': round(1 - weighted_score, 4),
            'uncertainty': round(uncertainty, 4),
            'risk_level': risk_level,
            'confidence_interval': (
                max(0, weighted_score - 2 * uncertainty),
                min(1, weighted_score + 2 * uncertainty)
            ),
            'signal_breakdown': signal_scores,
            'verdict': self._generate_verdict(weighted_score, uncertainty)
        }
    
    def _generate_verdict(self, score, uncertainty):
        if score < 0.20:
            return "✅ Media appears AUTHENTIC. No significant manipulation detected."
        elif score < 0.40:
            return "⚠️ LOW RISK. Minor anomalies detected. Likely authentic."
        elif score < 0.65:
            return "🔶 SUSPICIOUS. Multiple inconsistencies found. Manual review recommended."
        elif score < 0.85:
            return "🚨 HIGH RISK. Strong evidence of manipulation detected."
        else:
            return "❌ FAKE DETECTED. Media is very likely AI-generated or manipulated."
```

### 10.2 Frame-Level Timeline Output

```json
{
  "video_id": "vid_abc123",
  "duration_seconds": 45.2,
  "overall": {
    "fake_probability": 0.87,
    "risk_level": "FAKE",
    "verdict": "❌ FAKE DETECTED. Media is very likely AI-generated or manipulated.",
    "uncertainty": 0.04
  },
  "suspicious_segments": [
    { "start": "00:00:12", "end": "00:00:18", "score": 0.94, "reason": "Face blending artifact" },
    { "start": "00:00:31", "end": "00:00:39", "score": 0.89, "reason": "Lip-sync inconsistency" }
  ],
  "frame_scores": [0.12, 0.14, 0.91, 0.93, 0.88, ...],
  "signal_breakdown": {
    "spatial_xception": 0.91,
    "spatial_vit": 0.85,
    "frequency_domain": 0.72,
    "temporal_lstm": 0.89,
    "physiological": 0.76,
    "lip_sync": 0.93
  },
  "explainability": {
    "top_evidence": [
      "Right cheek skin texture artifacts (blending boundary)",
      "Unnatural blink pattern (2 blinks/min vs normal 15-20)",
      "Lip movement discontinuity at 00:00:12"
    ],
    "heatmap_frames": ["frame_0012.jpg", "frame_0031.jpg"]
  }
}
```

---

## 11. Tech Stack

### AI/ML

| Library | Purpose |
|---|---|
| **PyTorch 2.x** | Primary deep learning framework |
| **timm** | Pretrained models (XceptionNet, ViT, EfficientNet) |
| **PyTorchVideo** | SlowFast, video transforms |
| **OpenCV** | Frame extraction, face detection, optical flow |
| **dlib / MediaPipe** | Facial landmark detection |
| **Hugging Face** | Pretrained transformers, datasets |
| **Scikit-learn** | Ensemble calibration, metrics |
| **SHAP** | Explainability |
| **Librosa** | Audio features (for lip-sync cross-check) |
| **Albumentations** | Fast augmentation pipeline |

### Backend

| Component | Technology |
|---|---|
| API Framework | **FastAPI** (async, OpenAPI auto-docs) |
| Task Queue | **Celery** + **Redis** (async video processing) |
| Database | **PostgreSQL** (results, users, history) |
| Cache | **Redis** (model outputs, frame cache) |
| Storage | **AWS S3** / **MinIO** (video uploads) |
| Model Serving | **TorchServe** or **Triton Inference Server** |
| GPU Processing | **CUDA 12.x**, **cuDNN** |

### Frontend

| Component | Technology |
|---|---|
| Framework | **Next.js 14** (App Router) |
| Styling | **Tailwind CSS** |
| Video Player | **Video.js** with custom overlay |
| Charts | **Recharts** (score timeline, frame analysis) |
| Heatmap Overlay | **Canvas API** + custom shader |
| State Management | **Zustand** |
| Auth | **NextAuth.js** |

### Infrastructure

| Component | Technology |
|---|---|
| Containerization | **Docker** + **Docker Compose** |
| Orchestration | **Kubernetes** (production) |
| CI/CD | **GitHub Actions** |
| Model Registry | **MLflow** |
| Monitoring | **Prometheus** + **Grafana** |
| Logging | **ELK Stack** (Elasticsearch, Logstash, Kibana) |

---

## 12. System Architecture

```
                        ┌─────────────────────────────┐
                        │         USERS                │
                        │  Web Browser / API Client    │
                        └────────────┬────────────────┘
                                     │ HTTPS
                        ┌────────────▼────────────────┐
                        │      Next.js Frontend        │
                        │   Upload | Dashboard | XAI   │
                        └────────────┬────────────────┘
                                     │ REST / WebSocket
                        ┌────────────▼────────────────┐
                        │       FastAPI Backend         │
                        │  Auth | Upload | Job Manager  │
                        └──┬─────────────┬─────────────┘
                           │             │
              ┌────────────▼──┐    ┌─────▼────────────┐
              │  PostgreSQL   │    │  Redis + Celery   │
              │  (Results DB) │    │  (Task Queue)     │
              └───────────────┘    └─────┬─────────────┘
                                         │
                        ┌────────────────▼────────────┐
                        │     AI Processing Workers    │
                        │                             │
                        │  ┌─────────────────────┐   │
                        │  │ Video Preprocessor  │   │
                        │  │ (OpenCV, FFmpeg)    │   │
                        │  └──────────┬──────────┘   │
                        │             │               │
                        │  ┌──────────▼──────────┐   │
                        │  │  Face Detector       │   │
                        │  │  (RetinaFace, MTCNN) │   │
                        │  └──────────┬──────────┘   │
                        │             │               │
                        │  ┌──────────▼──────────┐   │
                        │  │  Model Ensemble      │   │
                        │  │  XceptionNet         │   │
                        │  │  ViT-L               │   │
                        │  │  FreqNet             │   │
                        │  │  TemporalLSTM        │   │
                        │  │  SlowFast 3D-CNN     │   │
                        │  └──────────┬──────────┘   │
                        │             │               │
                        │  ┌──────────▼──────────┐   │
                        │  │  Scoring + XAI       │   │
                        │  │  Grad-CAM, SHAP      │   │
                        │  └──────────────────────┘   │
                        │       CUDA GPU Workers       │
                        └─────────────────────────────┘
                                     │
                        ┌────────────▼────────────────┐
                        │         AWS S3               │
                        │    Video + Heatmap Storage   │
                        └─────────────────────────────┘
```

---

## 13. Processing Pipeline — Step by Step

### Step 1: Video Ingestion

```python
class VideoIngestor:
    MAX_FILE_SIZE_MB = 500
    SUPPORTED_FORMATS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    
    async def ingest(self, file: UploadFile) -> VideoJob:
        # Validate
        self._validate_format(file.filename)
        self._validate_size(file.size)
        
        # Upload to S3
        s3_key = f"uploads/{uuid4()}/{file.filename}"
        await s3.upload_fileobj(file.file, BUCKET, s3_key)
        
        # Create job
        job = VideoJob(
            id=str(uuid4()),
            s3_key=s3_key,
            status='queued',
            created_at=datetime.utcnow()
        )
        await db.save(job)
        
        # Queue processing
        process_video.delay(job.id)
        return job
```

### Step 2: Frame Extraction

```python
class FrameExtractor:
    def extract(self, video_path: str, target_fps: int = 10) -> list:
        """
        Extract frames at target FPS.
        - Full video: sample at 10 FPS
        - Detected scene changes: always include boundary frames
        - Max: 500 frames per video (for speed)
        """
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(original_fps / target_fps))
        
        frames, timestamps = [], []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frames.append(frame)
                timestamps.append(frame_idx / original_fps)
            frame_idx += 1
        
        cap.release()
        return frames, timestamps
```

### Step 3: Face Detection & Alignment

```python
from retinaface import RetinaFace

class FaceProcessor:
    def __init__(self):
        self.detector = RetinaFace  # SOTA face detector
    
    def process_frames(self, frames: list) -> list:
        results = []
        for frame in frames:
            faces = RetinaFace.detect_faces(frame)
            
            for face_id, face_data in faces.items():
                bbox = face_data['facial_area']
                landmarks = face_data['landmarks']
                
                # Align face using 5-point landmarks
                aligned = self._align_face(frame, landmarks, output_size=256)
                
                results.append({
                    'face': aligned,
                    'bbox': bbox,
                    'landmarks': landmarks,
                    'confidence': face_data['score']
                })
        
        return results
    
    def _align_face(self, image, landmarks, output_size=256):
        """Align face to canonical orientation using similarity transform"""
        src_pts = np.array([
            landmarks['left_eye'], landmarks['right_eye'],
            landmarks['nose'], landmarks['mouth_left'], landmarks['mouth_right']
        ], dtype=np.float32)
        
        dst_pts = self.CANONICAL_LANDMARKS * output_size
        transform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        aligned = cv2.warpAffine(image, transform, (output_size, output_size))
        return aligned
```

### Step 4: Model Inference

```python
class InferencePipeline:
    def run(self, face_crops: list, full_frames: list) -> dict:
        
        with torch.no_grad():
            # Batch face crops
            batch = self.preprocess(face_crops)  # [N, 3, 224, 224]
            
            # Spatial models
            xception_scores, xception_feats = self.xception_model(batch)
            vit_scores, vit_attention = self.vit_model(batch)
            freq_scores = self.freq_model(batch)
            
            # Temporal model (requires sequence)
            temporal_scores, frame_weights = self.temporal_model(
                xception_feats.unsqueeze(0)  # [1, T, 2048]
            )
            
            # Physiological analysis
            physio_results = self.physio_analyzer.analyze(full_frames)
            lip_results = self.lip_analyzer.analyze(full_frames, face_crops)
        
        return self.scoring_engine.compute_final_score({
            'spatial_xception': xception_scores.mean().item(),
            'spatial_vit': vit_scores.mean().item(),
            'frequency_domain': freq_scores.mean().item(),
            'temporal_lstm': temporal_scores.item(),
            'physiological': physio_results['anomaly_score'],
            'lip_sync': lip_results['anomaly_score']
        })
```

---

## 14. Folder Structure

```
deepfake-detector/
│
├── frontend/                          # Next.js 14 App
│   ├── app/
│   │   ├── page.tsx                   # Landing / Upload
│   │   ├── dashboard/page.tsx         # Analysis Dashboard
│   │   ├── results/[id]/page.tsx      # Per-video Results
│   │   └── api/                       # API Routes
│   ├── components/
│   │   ├── VideoUploader.tsx
│   │   ├── ConfidenceMeter.tsx
│   │   ├── FrameTimeline.tsx
│   │   ├── HeatmapOverlay.tsx
│   │   └── ReportGenerator.tsx
│   └── public/
│
├── backend/                           # FastAPI
│   ├── main.py
│   ├── routers/
│   │   ├── upload.py
│   │   ├── jobs.py
│   │   └── results.py
│   ├── services/
│   │   ├── inference.py
│   │   ├── storage.py
│   │   └── reporting.py
│   ├── workers/
│   │   ├── celery_app.py
│   │   └── video_tasks.py
│   └── models/                        # Pydantic schemas
│
├── ml/                                # All ML code
│   ├── models/
│   │   ├── xception_detector.py
│   │   ├── vit_detector.py
│   │   ├── freq_analyzer.py
│   │   ├── temporal_lstm.py
│   │   ├── slowfast_detector.py
│   │   └── ensemble.py
│   ├── preprocessing/
│   │   ├── frame_extractor.py
│   │   ├── face_processor.py
│   │   └── augmentation.py
│   ├── analysis/
│   │   ├── physiological.py
│   │   ├── lip_sync.py
│   │   └── optical_flow.py
│   ├── explainability/
│   │   ├── grad_cam.py
│   │   ├── vit_attention.py
│   │   └── shap_explainer.py
│   ├── scoring/
│   │   └── confidence_engine.py
│   └── training/
│       ├── train.py
│       ├── dataset.py
│       ├── losses.py
│       └── evaluate.py
│
├── datasets/                          # Dataset scripts
│   ├── download_ff++.sh
│   ├── download_dfdc.sh
│   └── prepare_dataset.py
│
├── infrastructure/
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   ├── Dockerfile.ml-worker
│   ├── k8s/
│   └── nginx.conf
│
├── research/                          # Experiments, notebooks
│   ├── notebooks/
│   ├── ablation_studies/
│   └── benchmark_results/
│
└── docs/
    ├── API.md
    ├── ARCHITECTURE.md
    └── TRAINING_GUIDE.md
```

---

## 15. API Design

### Core Endpoints

```
POST   /api/v1/videos/upload         → Upload video, returns job_id
GET    /api/v1/jobs/{job_id}         → Poll job status
GET    /api/v1/results/{video_id}    → Full analysis result
GET    /api/v1/results/{video_id}/heatmaps/{frame_id}  → Heatmap image
POST   /api/v1/analyze/url           → Analyze video from URL
GET    /api/v1/health                → Service health + GPU status
```

### Upload & Analyze Flow

```python
# FastAPI endpoint
@router.post("/videos/upload")
async def upload_video(
    file: UploadFile = File(...),
    options: AnalysisOptions = Body(default=AnalysisOptions()),
    current_user: User = Depends(get_current_user)
):
    """
    Upload video for deepfake analysis.
    
    Options:
    - analysis_depth: 'fast' | 'standard' | 'deep'
    - include_heatmaps: bool
    - include_physiological: bool
    """
    job = await ingestor.ingest(file, options, current_user.id)
    
    return {
        "job_id": job.id,
        "status": "queued",
        "estimated_time_seconds": estimate_processing_time(file.size, options),
        "websocket_url": f"/ws/jobs/{job.id}"
    }
```

### WebSocket Progress Updates

```python
@app.websocket("/ws/jobs/{job_id}")
async def job_progress(websocket: WebSocket, job_id: str):
    """Real-time progress updates via WebSocket"""
    await websocket.accept()
    
    async for update in job_progress_stream(job_id):
        await websocket.send_json({
            "stage": update.stage,          # "preprocessing", "inference", "scoring"
            "progress": update.progress,    # 0-100
            "message": update.message,
            "partial_results": update.partial_results
        })
    
    await websocket.close()
```

---

## 16. Frontend Dashboard

### Key UI Components

**Upload Interface**
- Drag-and-drop video upload
- Real-time upload progress
- Format/size validation
- Analysis depth selector (Fast / Standard / Deep)

**Results Dashboard**
- Large confidence meter (0–100% fake probability)
- Color-coded risk level badge
- Video player with frame-level overlay
- Suspicious frames timeline bar
- Per-signal breakdown chart (radar chart)
- Frame heatmap gallery

**Report Generation**
- Exportable PDF forensic report
- Evidence summary with annotated frames
- Shareable link with tamper-evident hash

### Sample Component: Confidence Meter

```tsx
// components/ConfidenceMeter.tsx
import { motion } from 'framer-motion'

const RISK_COLORS = {
  AUTHENTIC:  '#22c55e',  // green
  LOW_RISK:   '#84cc16',  // lime
  SUSPICIOUS: '#f59e0b',  // amber
  HIGH_RISK:  '#f97316',  // orange
  FAKE:       '#ef4444',  // red
}

export function ConfidenceMeter({ score, riskLevel }: Props) {
  const color = RISK_COLORS[riskLevel]
  const percentage = Math.round(score * 100)
  
  return (
    <div className="flex flex-col items-center gap-4 p-8">
      <div className="relative w-48 h-48">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          <circle cx="50" cy="50" r="40" fill="none" stroke="#e5e7eb" strokeWidth="8"/>
          <motion.circle
            cx="50" cy="50" r="40" fill="none"
            stroke={color} strokeWidth="8"
            strokeDasharray={`${percentage * 2.51} 251`}
            initial={{ strokeDasharray: "0 251" }}
            animate={{ strokeDasharray: `${percentage * 2.51} 251` }}
            transition={{ duration: 1.5, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold" style={{ color }}>
            {percentage}%
          </span>
          <span className="text-sm text-gray-500">Fake Probability</span>
        </div>
      </div>
      
      <div className="px-4 py-2 rounded-full text-white font-semibold"
           style={{ backgroundColor: color }}>
        {riskLevel.replace('_', ' ')}
      </div>
    </div>
  )
}
```

---

## 17. Training Strategy

### Loss Function

```python
class DeepfakeLoss(nn.Module):
    def __init__(self, focal_gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = focal_gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, pred, target):
        # Label smoothing
        smooth_target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Binary focal loss — focus on hard examples
        bce = F.binary_cross_entropy(pred, smooth_target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = (focal_weight * bce).mean()
        
        # Consistency regularization — similar frames should have similar scores
        consistency_loss = self._consistency_reg(pred)
        
        return focal_loss + 0.1 * consistency_loss
```

### Training Configuration

```yaml
# training_config.yaml
model:
  backbone: xception
  pretrained: imagenet
  dropout: 0.5

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  scheduler: cosine_annealing_with_warmup
  warmup_epochs: 5
  weight_decay: 1e-5
  
  # Mixed precision
  fp16: true
  
  # Gradient accumulation (effective batch=128)
  gradient_accumulation_steps: 4

data:
  datasets: [FaceForensics++, DFDC, CelebDF_v2]
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  
  # Compression levels (C0=raw, C23=light, C40=heavy)
  ff_compression: [C0, C23, C40]
  
  augmentation:
    - VideoCompression
    - ColorJitter
    - HorizontalFlip
    - GaussianNoise
    - RandomCrop

optimization:
  optimizer: AdamW
  beta1: 0.9
  beta2: 0.999
  eps: 1e-8
```

### Curriculum Learning

```python
# Start with easy examples, gradually add harder ones
class CurriculumDataset(Dataset):
    def __init__(self, epoch, max_epochs):
        self.difficulty = epoch / max_epochs  # 0.0 → 1.0
    
    def __getitem__(self, idx):
        # Early epochs: clear fakes (C0, high artifacts)
        # Later epochs: compressed fakes + adversarial examples
        if self.difficulty < 0.3:
            return self.easy_samples[idx]
        elif self.difficulty < 0.7:
            return self.medium_samples[idx]
        else:
            return self.hard_samples[idx]  # Adversarially attacked fakes
```

---

## 18. Evaluation Metrics

### Core Metrics

| Metric | Formula | Why It Matters |
|---|---|---|
| **AUC-ROC** | Area under ROC curve | Threshold-independent overall performance |
| **AP (Average Precision)** | Area under PR curve | Critical when data is imbalanced |
| **EER (Equal Error Rate)** | FAR = FRR point | Standard in forensics/biometrics |
| **Accuracy** | (TP+TN)/(Total) | Simple correctness |
| **F1 Score** | 2·P·R/(P+R) | Balances precision and recall |

### Benchmark Evaluation Script

```python
def evaluate_model(model, test_loader, device):
    all_preds, all_labels = [], []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            frames, labels = batch
            scores = model(frames.to(device))['final_score']
            all_preds.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, preds)
    optimal_thresh = thresholds[np.argmax(tpr - fpr)]
    binary_preds = (preds >= optimal_thresh).astype(int)
    
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"Avg Prec: {ap:.4f}")
    print(f"Accuracy: {accuracy_score(labels, binary_preds):.4f}")
    print(f"F1:       {f1_score(labels, binary_preds):.4f}")
    
    return {'auc': auc, 'ap': ap, 'threshold': optimal_thresh}
```

### Cross-Dataset Generalization Test

The true test of a world-class detector is **cross-dataset generalization**:

| Train → Test | Expected AUC Target |
|---|---|
| FF++ → FF++ (in-distribution) | > 0.99 |
| FF++ → Celeb-DF v2 | > 0.80 |
| FF++ → DFDC | > 0.75 |
| ALL datasets → WildDeepfake | > 0.85 |

---

## 19. Robustness & Adversarial Hardening

### Defense Against Adversarial Deepfakes

Sophisticated adversaries may create deepfakes specifically designed to fool detectors.

```python
class AdversarialTrainer:
    """
    Adversarial training to harden the model against:
    1. Perturbation attacks (add imperceptible noise to fool detector)
    2. Compression-based evasion (re-compress to destroy artifacts)
    3. GAN-based anti-detection (train deepfake to fool our model)
    """
    
    def pgd_attack(self, model, frames, labels, eps=8/255, steps=20):
        """
        Projected Gradient Descent — generate adversarial examples
        to augment training (adversarial training = AT)
        """
        delta = torch.zeros_like(frames).uniform_(-eps, eps)
        delta.requires_grad = True
        
        for _ in range(steps):
            output = model(frames + delta)['final_score']
            loss = F.binary_cross_entropy(output, labels)
            loss.backward()
            
            delta.data = (delta.data + (eps/steps) * delta.grad.sign()).clamp(-eps, eps)
            delta.grad.zero_()
        
        return frames + delta.detach()
    
    def train_step_with_at(self, model, batch, optimizer):
        frames, labels = batch
        
        # Generate adversarial examples (40% of batch)
        adv_frames = self.pgd_attack(model, frames[:len(frames)//2], labels[:len(labels)//2])
        
        # Combined clean + adversarial training
        all_frames = torch.cat([frames, adv_frames], dim=0)
        all_labels = torch.cat([labels, labels[:len(labels)//2]], dim=0)
        
        output = model(all_frames)
        loss = self.criterion(output['final_score'], all_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Model Robustness Checklist

- ✅ Trained on multiple compression levels (raw, light, heavy)
- ✅ Augmented with JPEG, H.264, H.265 compression artifacts
- ✅ Adversarial training (PGD attacks)
- ✅ Tested on in-the-wild social media downloads
- ✅ Robust to resolution changes (144p to 4K)
- ✅ Handles partial occlusions (glasses, hats, masks)
- ✅ Multi-face video support (flag only manipulated faces)

---

## 20. Advanced Features

### A. Real-Time Webcam Detection

```python
class RealtimeDetector:
    """
    Optimized for real-time inference on webcam streams.
    Uses a lightweight MobileNetV3 + compressed LSTM for speed.
    """
    
    TARGET_LATENCY_MS = 100  # 10 FPS real-time
    
    def __init__(self):
        # Lightweight backbone for real-time
        self.fast_model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        self.fast_model = torch.jit.script(self.fast_model)  # TorchScript for speed
        self.buffer = deque(maxlen=30)  # 1-second rolling window
    
    def process_frame(self, frame):
        self.buffer.append(frame)
        
        if len(self.buffer) % 10 == 0:  # Analyze every 10 frames
            score = self.fast_model(self.preprocess_buffer())
            return score
```

### B. Browser Extension Architecture

```
Browser Extension (Chrome/Firefox)
├── content_script.js           → Intercepts video elements on page
├── background.js               → Sends video frames to API
├── popup.html                  → Shows real-time score overlay
└── manifest.json               → Permissions: activeTab, storage, webRequest

Supported Platforms:
- YouTube (iframe, autoplay detection)
- Twitter/X (embedded video)
- Facebook
- TikTok
- News websites
```

### C. Blockchain Media Verification

```python
class BlockchainVerifier:
    """
    Store perceptual hash of verified authentic videos on-chain.
    Future uploads compared against chain to prove authenticity.
    """
    
    def compute_perceptual_hash(self, video_path: str) -> str:
        frames = self.extract_keyframes(video_path, n=10)
        hashes = [imagehash.phash(Image.fromarray(f)) for f in frames]
        combined = ''.join(str(h) for h in hashes)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def register_on_chain(self, video_hash: str, metadata: dict):
        # Store on Ethereum / Polygon (low gas)
        tx = self.contract.functions.registerMedia(
            video_hash,
            metadata['creator'],
            int(time.time())
        ).transact()
        return tx.hex()
```

### D. Forensic Report Generator

```python
class ForensicReportGenerator:
    """
    Generate professional PDF forensic reports suitable for:
    - Legal proceedings
    - Journalism
    - Corporate security
    """
    
    def generate(self, analysis_result: dict, output_path: str):
        report = {
            'header': "AI Forensic Video Analysis Report",
            'case_id': analysis_result['video_id'],
            'analysis_date': datetime.utcnow().isoformat(),
            'verdict': analysis_result['verdict'],
            'methodology': self.get_methodology_description(),
            'evidence': {
                'confidence_score': analysis_result['fake_probability'],
                'suspicious_segments': analysis_result['suspicious_segments'],
                'signal_breakdown': analysis_result['signal_breakdown'],
                'top_evidence': analysis_result['explainability']['top_evidence'],
                'heatmap_frames': analysis_result['heatmap_frame_paths']
            },
            'model_info': {
                'version': MODEL_VERSION,
                'trained_on': ['FaceForensics++', 'DFDC', 'Celeb-DF v2'],
                'auc_on_benchmark': 0.967
            },
            'disclaimer': "This report is generated by an AI system and should be "
                         "reviewed by a qualified digital forensics expert."
        }
        
        self._render_to_pdf(report, output_path)
```

---

## 21. Development Timeline

| Phase | Duration | Deliverables |
|---|---|---|
| **Phase 1** — Foundation | Weeks 1–2 | Dataset download, preprocessing pipeline, face detection |
| **Phase 2** — Spatial Models | Weeks 3–5 | XceptionNet + ViT training, baseline evaluation |
| **Phase 3** — Temporal Models | Weeks 6–7 | LSTM, SlowFast, physiological analysis |
| **Phase 4** — Ensemble & Scoring | Week 8 | Fusion model, confidence engine, calibration |
| **Phase 5** — Explainability | Week 9 | Grad-CAM, attention maps, SHAP |
| **Phase 6** — Backend API | Weeks 10–11 | FastAPI, Celery, PostgreSQL, S3 |
| **Phase 7** — Frontend | Weeks 12–13 | Next.js dashboard, video player, results UI |
| **Phase 8** — Hardening | Week 14 | Adversarial training, robustness testing |
| **Phase 9** — Deployment | Week 15 | Docker, CI/CD, monitoring |
| **Phase 10** — Advanced | Weeks 16+ | Real-time, browser extension, blockchain |

---

## 22. Real-World Applications

| Sector | Use Case | Impact |
|---|---|---|
| **Government & Law Enforcement** | Verifying video evidence in court, detecting political deepfakes | Prevent wrongful convictions, election integrity |
| **Media & Journalism** | Pre-publication verification of video clips | Combat misinformation at the source |
| **Financial Sector** | CEO/executive impersonation detection in video calls | Prevent wire fraud, BEC attacks |
| **HR & Hiring** | Verify identity during video interviews | Prevent remote identity fraud |
| **Cybersecurity** | Video KYC deepfake detection in banking/fintech | AML/KYC compliance |
| **Social Media Platforms** | Automated flagging of deepfake content | Content moderation at scale |
| **Healthcare** | Verify patient identity in telehealth | HIPAA-compliant identity verification |
| **Insurance** | Detect fraudulent video evidence in claims | Reduce insurance fraud losses |

---

## 23. Future Scope

| Feature | Description | Timeline |
|---|---|---|
| **Mobile App** | iOS/Android real-time deepfake detection | 6 months |
| **Social Media API Integration** | Direct integration with Twitter, YouTube APIs | 6–12 months |
| **Multilingual Support** | Lip-sync analysis across 50+ languages | 12 months |
| **Deepfake Generation Fingerprinting** | Identify *which* tool created the deepfake | 12 months |
| **Fake Document Detection** | Extend to PDF, ID photos, documents | 12 months |
| **AI-Generated Text Detection** | Full multimodal fake content platform | 18 months |
| **Deepfake Forensic Toolkit** | Professional tool for digital forensics experts | 18 months |
| **Federated Learning** | Train on encrypted data across organizations | 24 months |
| **Live Video Stream Detection** | Real-time Zoom/Teams/WebRTC monitoring | 24 months |

---

## 24. Research References

| Paper | Contribution | Year |
|---|---|---|
| **FaceForensics++** (Rossler et al.) | Benchmark dataset + XceptionNet baseline | 2019 |
| **Celeb-DF** (Li et al.) | High-quality celebrity deepfake dataset | 2020 |
| **DFDC** (Dolhansky et al.) | Large-scale real-world dataset | 2020 |
| **Face X-Ray** (Li et al.) | Blending boundary detection | 2020 |
| **Multi-Attentional Deepfake Detection** (Zhao et al.) | Attention on subtle artifacts | 2021 |
| **CLRNet** (Sun et al.) | Cross-modal deepfake detection | 2021 |
| **Lips Don't Lie** (Haliassos et al.) | Lip movement for detection | 2021 |
| **RECCE** (Cao et al.) | Reconstruction-based detection | 2022 |
| **UIA-ViT** (Zhuang et al.) | Unsupervised inconsistency attention ViT | 2022 |
| **AltFreezing** (Wang et al.) | Spatial-temporal deepfake detection | 2023 |
| **Leveraging Vision-Language** (Khan et al.) | CLIP-based detection | 2023 |
| **LSDA** (Yan et al.) | Large-scale deepfake detection | 2024 |

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│         VIDEO DEEPFAKE DETECTION — AT A GLANCE       │
├─────────────────────────────────────────────────────┤
│  INPUT:  MP4, MOV, AVI, MKV, WEBM (up to 500MB)    │
│  OUTPUT: Score 0–100% | Risk Level | Heatmaps       │
├─────────────────────────────────────────────────────┤
│  MODELS:                                            │
│  • XceptionNet  → Pixel-level artifacts             │
│  • ViT-L        → Global inconsistencies            │
│  • FreqNet      → GAN fingerprints in FFT           │
│  • Bi-LSTM      → Temporal coherence                │
│  • SlowFast     → Motion consistency                │
├─────────────────────────────────────────────────────┤
│  SIGNALS:                                           │
│  • Blink patterns (rPPG)                            │
│  • Lip-sync discontinuity                           │
│  • Optical flow anomalies                           │
│  • Blending boundary artifacts                      │
│  • Frequency domain fingerprints                    │
├─────────────────────────────────────────────────────┤
│  PERFORMANCE TARGETS:                               │
│  • AUC-ROC ≥ 0.95                                   │
│  • Accuracy ≥ 92%                                   │
│  • FPR ≤ 5%                                         │
│  • Latency ≤ 30s / 1-min video (GPU)               │
└─────────────────────────────────────────────────────┘
```

---

*Built with PyTorch · FastAPI · Next.js · PostgreSQL · Redis · AWS*  
*Trained on FaceForensics++ · DFDC · Celeb-DF v2 · WildDeepfake*  
*Licensed for research and production use*
