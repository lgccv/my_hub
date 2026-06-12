import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box


class YOLOClsEmbedding(nn.Module):
    """导出专用：返回和 ReID(embed=倒数第二层)一致的 embedding。"""

    def __init__(self, model_path: str, embed_index: int | None = None):
        super().__init__()
        yolo = YOLO(model_path)
        self.model = yolo.model
        self.model.eval()

        modules = self.model.model
        self.embed_index = len(modules) - 2 if embed_index is None else embed_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = []
        for m in self.model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.model.save else None)

            if m.i == self.embed_index:
                return F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

        raise RuntimeError(f"embed layer {self.embed_index} not reached")


def export_embedding_onnx(
    pt_path: str = "/home/std/workspace-hub/lgc/yolov11/yolo11n-cls.pt",
    onnx_path: str = "/home/std/workspace-hub/lgc/yolov11/yolo11n-cls-embed.onnx",
):
    model = YOLOClsEmbedding(pt_path)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["embeddings"],
        opset_version=17,
    )


class ReIDPT:
    """你的 PT 版本，保留原语义。"""

    def __init__(self, model: str):
        self.model = YOLO(model)
        self.model(embed=[len(self.model.model.model) - 2], verbose=False, save=False)

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        feats = self.model.predictor(
            [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        )
        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]
        return [f.cpu().numpy() for f in feats]


class Preprocess:
    def __init__(self,img_size):
        self.imgsz = img_size
        
    def __call__(self,crops,mean=(0,0,0),std=(1,1,1)):
        batch = []
        for crop in crops:
            if crop is None or crop.size == 0:
                raise ValueError("Empty crop found.")

            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            if h < w:
                new_h = self.imgsz
                new_w = int(round(w * self.imgsz / h))
            else:
                new_w = self.imgsz
                new_h = int(round(h * self.imgsz / w))

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            top = (new_h - self.imgsz) // 2
            left = (new_w - self.imgsz) // 2
            img = img[top:top + self.imgsz, left:left + self.imgsz]

            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            img = (img - mean[:, None, None]) / std[:, None, None]

            batch.append(img)

        batch = np.stack(batch, axis=0).astype(np.float32)
        return batch


class ReIDONNX:
    def __init__(self,model_path,imgsz=224,use_cuda=True):
        providers = ["CPUExecutionProvider"]
        if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.preprocess = Preprocess(imgsz)

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        if len(dets) == 0:
            return []

        dets[:,2:] = dets[:,2:]*1.02+10
        h, w = img.shape[:2]
        crops = []
        for det in dets:
            y1 = int(det[1] - det[3]/2)
            y2 = int(det[1] + det[3]/2)
            x1 = int(det[0] - det[2]/2)
            x2 = int(det[0] + det[2]/2)
            crops.append(img[y1:y2, x1:x2].copy())

        batch = self.preprocess(crops)
        feats = self.session.run([self.output_name], {self.input_name: batch})[0]
        return [feat.astype(np.float32) for feat in feats]


        

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


if __name__ == "__main__":
    pt_path = "/home/std/workspace-hub/lgc/yolov11/yolo11n-cls.pt"
    onnx_path = "/home/std/workspace-hub/lgc/yolov11/yolo11n-cls-embed.onnx"
    img_path = "/home/std/workspace-hub/lgc/yolov11/images/images_0401/1774939924_914488064_369.jpg"

    # 第一次先导出
    export_embedding_onnx(pt_path, onnx_path)

    image = cv2.imread(img_path)

    # 注意：这里必须是 xywh，不是 xyxy
    dets = np.array([
        [150, 150, 100, 100],
    ], dtype=np.float32)

    model_pt = ReIDPT(pt_path)
    feat_pt = model_pt(image, dets)[0]
    print("feat_pt",feat_pt)

    model_onnx = ReIDONNX(onnx_path)
    feat_opencv = model_onnx(image, dets)[0]
    print("feat_opencv",feat_opencv)

    cos = cosine_similarity(feat_pt,feat_opencv)

    print("cos:",cos)
