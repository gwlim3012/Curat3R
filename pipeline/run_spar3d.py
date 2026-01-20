import argparse
import os
import torch
import sys
import gc
from PIL import Image

# [메모리 최적화]
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# [Hugging Face 토큰 설정]
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
os.environ["HF_TOKEN"] = hf_token
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from spar3d.system import SPAR3D
    from transparent_background import Remover
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from spar3d.system import SPAR3D
    from transparent_background import Remover

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    
    # ✅ [컨트롤 타워] 모든 설정을 여기서 관리
    parser.add_argument("--texture-resolution", type=int, default=4096) # 해상도
    parser.add_argument("--remesh_option", type=str, default="triangle")
    parser.add_argument("--reduction_count_type", type=str, default="vertex")
    parser.add_argument("--target_count", type=int, default=50000) # 점 개수

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"[SPAR3D] Loading Model... (Res: {args.texture_resolution}, Verts: {args.target_count})")
    
    device = torch.device(args.device)
    
    # 배경 제거기 로드
    print(f"[SPAR3D] Removing Background...")
    remover = Remover(mode='base', device='cuda')
    
    # 3D 모델 로드
    model = SPAR3D.from_pretrained(
        "stabilityai/stable-point-aware-3d",
        config_name="config.yaml",
        weight_name="model.safetensors"
    ).to(device)
    model.eval()
    
    print(f"[SPAR3D] Processing Image from {args.image_path}...")
    
    # 배경 제거 실행
    original_image = Image.open(args.image_path).convert("RGB")
    input_image = remover.process(original_image, type='rgba')
    
    # 옵션 매핑
    if args.reduction_count_type == "vertex":
        vertex_count = args.target_count
    elif args.reduction_count_type == "faces":
        vertex_count = args.target_count // 2
    else:
        vertex_count = -1 

    print(f"[SPAR3D] Generating 3D Mesh...")
    
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda'): 
                result = model.run_image(
                    input_image, 
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh_option,
                    vertex_count=vertex_count
                )
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("⚠️ OOM Error! Still running out of memory.")
            torch.cuda.empty_cache()
            raise e
        else:
            raise e
    
    if isinstance(result, tuple):
        mesh = result[0]
    else:
        mesh = result
    
    save_path = os.path.join(args.output_dir, "mesh.glb")
    mesh.export(save_path)
    print(f"[SPAR3D] Success! Saved to {save_path}")

if __name__ == "__main__":
    main()