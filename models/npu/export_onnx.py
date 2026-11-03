"""Export the Stage 1 keypoint extractor to ONNX for the Mobilint Qubee compiler.

    python models/npu/export_onnx.py --checkpoint runs/keypoints_resnet50_256x192/best.pt --out_dir npu_export

Writes three files:
  model.onnx           full network: (1, 256, 192, 3) RGB in [0, 1] -> (1, 73, 64, 48) heatmaps
  model_backbone.onnx  plan B if Qubee rejects ConvTranspose: the same input -> ResNet features
  model_head.onnx      the deconv head for plan B, run on the CPU with onnxruntime

The input is NHWC with values in [0, 1], the layout the MACCEL runtime takes in the labs, and the
ImageNet normalisation is inside the graph, so the NPU sees plain pixels.
"""
import argparse
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "keypoints"))

from predict_wlasl import load_model  # noqa: E402
from transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


class NHWCInput(nn.Module):
    """(1, H, W, 3) in [0, 1] -> ImageNet-normalised NCHW, then the wrapped module."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("scale", torch.from_numpy(1.0 / IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("shift", torch.from_numpy(-IMAGENET_MEAN / IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2) * self.scale + self.shift)


def fold_deconv_batchnorm(model):
    """Fold each BatchNorm after a ConvTranspose into its weights and bias. The exporter only folds
    BatchNorm into Conv, so without this the head reaches the compiler as ConvTranspose + BatchNormalization."""
    layers = list(model.deconv)
    folded = []
    for deconv, bn, relu in zip(layers[0::3], layers[1::3], layers[2::3]):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused = nn.ConvTranspose2d(deconv.in_channels, deconv.out_channels, deconv.kernel_size, stride=deconv.stride,
                                   padding=deconv.padding, bias=True)
        with torch.no_grad():
            fused.weight.copy_(deconv.weight * scale.view(1, -1, 1, 1))  # ConvTranspose weight is (in, out, kh, kw)
            fused.bias.copy_(bn.bias - bn.running_mean * scale)
        folded += [fused, relu]
    model.deconv = nn.Sequential(*folded)
    return model


class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.deconv, self.head = model.deconv, model.head

    def forward(self, x):
        return self.head(self.deconv(x))


def export(module, dummy, path, opset):
    torch.onnx.export(module, dummy, path, export_params=True, opset_version=opset, do_constant_folding=True,
                      input_names=["input"], output_names=["output"], dynamic_axes=None, dynamo=False)
    m = onnx.load(path)
    # The exporter shares identical constants through Identity nodes; point their users at the constant instead
    # (same clean-up as the Lab 2 export).
    alias = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "Identity"}
    if alias:
        for n in m.graph.node:
            for k, name in enumerate(n.input):
                if name in alias:
                    n.input[k] = alias[name]
        keep = [n for n in m.graph.node if n.op_type != "Identity"]
        del m.graph.node[:]
        m.graph.node.extend(keep)
        onnx.save(m, path)
    onnx.checker.check_model(m)
    shape = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    print(f"[export] {os.path.basename(path)}: {shape(m.graph.input[0])} -> {shape(m.graph.output[0])}, "
          f"ops {sorted({n.op_type for n in m.graph.node})}")


def main():
    parser = argparse.ArgumentParser(description="Export the Stage 1 extractor to ONNX for the Mobilint NPU.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, train_args = load_model(args.checkpoint, torch.device("cpu"))
    h, w = train_args["input_h"], train_args["input_w"]
    image = torch.rand(1, h, w, 3)
    with torch.no_grad():
        reference = NHWCInput(model).eval()(image)
    model = fold_deconv_batchnorm(model).eval()

    full = NHWCInput(model).eval()
    backbone = NHWCInput(model.backbone).eval()
    head = Head(model).eval()
    with torch.no_grad():
        features = backbone(image)
        # Folding and splitting must both reproduce the original network.
        assert torch.allclose(full(image), reference, atol=1e-4)
        assert torch.allclose(head(features), full(image), atol=1e-5)

    export(full, image, os.path.join(args.out_dir, "model.onnx"), args.opset)
    export(backbone, image, os.path.join(args.out_dir, "model_backbone.onnx"), args.opset)
    export(head, features, os.path.join(args.out_dir, "model_head.onnx"), args.opset)

    import onnxruntime as ort
    session = ort.InferenceSession(os.path.join(args.out_dir, "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": image.numpy()})[0]
    with torch.no_grad():
        torch_out = full(image).numpy()
    print(f"[export] max |ONNX - PyTorch| on a random input: {np.abs(onnx_out - torch_out).max():.2e}")


if __name__ == "__main__":
    main()
