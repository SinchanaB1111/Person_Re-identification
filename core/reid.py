"""Person Re-ID feature extractor (OSNet-AIN x1.0 trained on MSMT17).

Turns a person crop into a 512-d L2-normalised appearance vector. Two crops of the
same person, even from different cameras, should have a high cosine similarity.

On CPU the network runs through ONNX Runtime (about 5x faster than PyTorch for this
model, same outputs). The PyTorch weights are the fallback and the source of the
ONNX file: `python -m core.reid` rebuilds it.
"""
import cv2
import numpy as np
import torch

from .config import REID_ONNX, REID_WEIGHTS, get_device
from .osnet_ain import osnet_ain_x1_0

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_H, INPUT_W = 256, 128


def _load_torch_model(weights=REID_WEIGHTS):
    state = torch.load(weights, map_location="cpu")
    state = state.get("state_dict", state)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model = osnet_ain_x1_0(num_classes=state["classifier.weight"].shape[0])
    model.load_state_dict(state)
    return model.eval()


class ReIDModel:
    def __init__(self, weights=REID_WEIGHTS, device=None):
        self.device = device or get_device()
        self.session = None
        if self.device == "cpu" and REID_ONNX.exists():
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(REID_ONNX), providers=["CPUExecutionProvider"])
            return
        self.model = _load_torch_model(weights).to(self.device)

    def _preprocess(self, crops_bgr):
        batch = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            rgb = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
            batch.append(rgb.transpose(2, 0, 1))
        return torch.from_numpy(np.stack(batch))

    @torch.inference_mode()
    def extract(self, crops_bgr, batch_size=32):
        """List of BGR crops -> (N, 512) float32 array, L2-normalised."""
        crops_bgr = [c for c in crops_bgr]
        if not crops_bgr:
            return np.zeros((0, 512), dtype=np.float32)
        out = []
        for i in range(0, len(crops_bgr), batch_size):
            x = self._preprocess(crops_bgr[i:i + batch_size])
            if self.session is not None:
                f = self.session.run(None, {"x": x.numpy()})[0]
                f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
            else:
                f = torch.nn.functional.normalize(self.model(x.to(self.device)), dim=1).cpu().numpy()
            out.append(f)
        return np.concatenate(out).astype(np.float32)


def export_onnx():
    model = _load_torch_model()
    torch.onnx.export(model, torch.randn(1, 3, INPUT_H, INPUT_W), str(REID_ONNX), input_names=["x"],
                      output_names=["f"], dynamic_axes={"x": {0: "n"}, "f": {0: "n"}}, opset_version=17,
                      dynamo=False)
    print(f"wrote {REID_ONNX}")


if __name__ == "__main__":
    export_onnx()
