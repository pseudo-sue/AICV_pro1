import argparse, os, glob, sys, time
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont

# --- 필수: 프로젝트 루트를 PYTHONPATH에 추가 (src.* 임포트용) ---
ROOT = Path(__file__).resolve().parents[1]  # .../rtdetrv2_pytorch
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig  # 프로젝트 내부 유틸

KITTI_CLASSES = [
    "Car","Van","Truck","Pedestrian","Person_sitting",
    "Cyclist","Tram"
]

def vis_one(img_pil, boxes, scores, labels, class_names, thr=0.5):
    im = img_pil.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    keep = (scores >= thr)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    for b, s, l in zip(boxes, scores, labels):
        x1,y1,x2,y2 = [float(v) for v in b.tolist()]
        cls_id = int(l.item())
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls{cls_id}"
        draw.rectangle([(x1,y1),(x2,y2)], outline="red", width=3)
        text = f"{name}:{s:.2f}"
        tw, th = draw.textlength(text, font=font), 16
        draw.rectangle([(x1,y1-th-2),(x1+tw+6,y1)], fill="red")
        draw.text((x1+3,y1-th-2), text, fill="white", font=font)
    return im

def build_val_transform_from_cfg(cfg):
    # cfg의 val transforms를 간단 매핑 (Resize(640,640) + ToTensor(0~1))로 축약
    # RT-DETR의 postprocessor는 orig_target_sizes를 받아 scale 보정하므로 normalize 불필요
    from torchvision import transforms as T
    return T.Compose([T.Resize((640,640)), T.ToTensor()])

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("-r","--weights", required=True)
    ap.add_argument("--im-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--classes", default="kitti", choices=["kitti","coco"])
    args = ap.parse_args()

    class_names = KITTI_CLASSES if args.classes == "kitti" else None  # coco는 repo 기본 매핑 사용 가능 시 대체

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = YAMLConfig(args.config)

    # 모델/후처리 구성 요소 로드
    model = cfg.model.to(device).eval()
    post = cfg.postprocessor

    # 가중치 로드
    state = torch.load(args.weights, map_location="cpu")
    # checkpoint 형태 다양성 대응
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)

    # 변환 정의
    tfm = build_val_transform_from_cfg(cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    exts = ("*.jpg","*.png","*.jpeg","*.bmp")
    images = []
    for e in exts:
        images += glob.glob(os.path.join(args.im_dir, e))
    images = sorted(images)

    total = 0.0
    for p in images:
        img = Image.open(p).convert("RGB")
        w, h = img.size

        x = tfm(img).unsqueeze(0).to(device)  # [1,3,640,640]
        orig = torch.tensor([[w, h]], dtype=torch.int64, device=device)

        t0 = time.time()
        out = model(x)
        # out: dict with "pred_logits"(B,Q,C), "pred_boxes"(B,Q,4 in cxcywh normalized)
        # repo의 postprocessor가 (out, orig_sizes) -> list[dict(boxes,labels,scores)]
        det = post(out, orig)[0]
        total += (time.time() - t0)

        boxes = det["boxes"].cpu()     # [N,4] xyxy
        scores = det["scores"].cpu()   # [N]
        labels = det["labels"].cpu()-1   # [N]

        # 클래스 이름 결정
        names = class_names
        if names is None:
            # fallback: 숫자 라벨만 표기
            names = [f"cls{i}" for i in range(int(labels.max().item()+1) if labels.numel() else 1)]

        vis = vis_one(img, boxes, scores, labels, names, thr=args.thr)
        out_path = os.path.join(args.out_dir, Path(p).stem + "_pred.jpg")
        vis.save(out_path)

    if images:
        print(f"Done. Saved {len(images)} results to: {args.out_dir}")
        print(f"Avg latency per image: {total/len(images)*1000:.1f} ms")
    else:
        print("No images found.")

if __name__ == "__main__":
    main()