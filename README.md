# Curat3R — My 3D Museum

A simple web service for preserving digital artifacts by producing 3D reconstructions from a single image.

*ToBig Conference CV Project*

## 👥 Project Members

| Name | Affiliation |
|---|---|
| Minjung Kim | ToBigs 23rd |
| Sangwoo Kim | ToBigs 23rd |
| Jungwoo Yoon | ToBigs 24th |
| Gyuwon Lim | ToBigs 23rd |
| Seoyoung Jeong | ToBigs 23rd |

---

## 📁 Directory Structure

```
Curat3R/
├─ pipeline/                 # Pipeline server (Flask) and helper scripts
│  ├─ pipeline_server.py     # Flask entry (CLIP filtering + SF3D/Trellis reconstruction)
│  ├─ clip_filter.py         # CLIP-based image filter
│  ├─ run_trellis.py         # Trellis helper script
│  ├─ requirements.txt       # Python dependencies (minimal)
│  ├─ models/                # Model metadata/config (use Git LFS for large weights)
│  └─ start_server.sh        # Helpful start script (Linux/WSL recommended)

├─ src/                      # Next.js web application (UI)
│  ├─ app/                   # Pages / routes
│  ├─ components/            # React components (ModelViewer, Thumbnail, etc.)
│  └─ services/              # Frontend services (pipeline API client)

└─ README.md                 # (this file)
```

---

## ✨ Key Features

- CLIP-based filtering (`clip_filter.py`)
  - Classifies whether an image is suitable for 3D reconstruction (accept / reject).
- 3D Reconstruction
  - StableFast3D (fast mode)
  - Trellis (quality mode — requires separate TRELLIS.2 repo)
- Web UI (Next.js)
  - Upload images, display filter status, download 3D models, and generate thumbnails

---

## ⚙️ Getting Started (local development)

> On Windows we recommend using WSL or Docker. Many components assume Linux paths and CUDA-enabled GPUs; running natively on Windows may require path and environment adjustments.

### ✅ Verified environment (tested)

- GPU: NVIDIA RTX 5090
- CUDA: 12.9
- Recommended VRAM: 32 GB (for smoother, higher-quality reconstructions)

Note: Ensure NVIDIA drivers and the PyTorch/CUDA build match your CUDA version.

### 1) Frontend (Next.js)

```bash
cd src
npm install
npm run dev
# Open: http://localhost:3000
```

### 2) Pipeline Server (Flask)

```bash
cd pipeline
# (recommended) Linux / WSL
python -m venv .venv
source .venv/bin/activate       # Windows Powershell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
# Note: Install PyTorch separately for your platform (CUDA or CPU)
# Example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Run server
python pipeline_server.py
# Or use helper script
./start_server.sh
```

### 3) Pipeline API examples

- Filter only
  - POST `/api/pipeline/filter` (form-data: `image`)
- Full pipeline (filter + reconstruction)
  - POST `/api/pipeline/process` (form-data: `image`)
- Reconstruct only (skip filtering)
  - POST `/api/pipeline/reconstruct/<task_id>` (JSON: { "model": "fast" | "quality" })
- Download result
  - GET `/api/pipeline/download/<task_id>`

---

## 🧩 Required External Resources / Models

- **CLIP** (ViT-B/32)
  - The `clip` Python package downloads the model automatically via `clip.load`.
- **StableFast3D**
  - Requires the StableFast3D repository or server. Verify `SF3D_ENV`, `SF3D_SCRIPT`, and `SF3D_LOCAL_MODEL` paths in `pipeline_server.py`.
  - Put large model weights (`.safetensors`) under `pipeline/models/` or obtain them via the StableFast3D repo.
- **Trellis (Quality mode)**
  - Requires the TRELLIS.2 repository and a Conda environment (e.g., `trellis311`).

### Environment variables

- `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` — set if you need to access gated Hugging Face models or private assets.

### Hardcoded paths

- Note: Several scripts (e.g., `pipeline_server.py`, `run_trellis.py`) contain hardcoded Linux paths like `/workspace/tobigs/...`. Update these paths for your local environment (Windows/WSL/Docker) as needed.

---
