# fah-metal

[FAH](https://github.com/noahcao/FAH) hand pose on Apple silicon, with the network rewritten in plain Metal compute.
Per hand: 21 joints in 2D / 3D, a MANO mesh, MANO rotations and FAH's hand-object contact. **115-234 fps** on an M5.

> **FAH is the work of Jinkun Cao**: *Fast and Accurate Hand Estimation in Pixels, Interaction and World Space*, 2026
> ([arXiv 2611.07711](https://arxiv.org/abs/2611.07711), [code](https://github.com/noahcao/FAH),
> [project page](https://noahcao.github.io/FAH/)). This repo only ports its inference. Please cite the paper.

## Results

Apple M5 (10-core GPU), macOS 26.7, medians.

**Whole frame** (`fah-check speed`: 120 frames of a 1440 × 1920 video, one hand, tracking):

| Mesh | Contact | ms / frame | fps |
|---|---|---|---|
| off | off | 4.3 | **234** |
| on | off | 5.4 | **184** |
| off | on | 7.4 | **136** |
| on | on | 8.7 | **115** |

**Pose network**, one 320 × 320 crop:

| | Batch 1 | Batch 2 |
|---|---|---|
| **Metal, this repo** | **4.0 ms** | **7.5 ms** |
| MLX fp16, same graph | 10.8 ms | 17 ms |
| Core ML fp32 (FAH's export) | 12.5 ms | 23.5 ms |
| Metal, simdgroup fallback (no neural accelerators) | 10.7 ms | 20 ms |
| + contact head | +3 ms | +6 ms |

**Accuracy**:

| | vs | Result |
|---|---|---|
| Pose network | PyTorch fp32, 16 crops | uv 0.054 px median (MLX fp16: 0.116), xyz 0.056 mm |
| Pipeline | Python FAH, 412 frames | same detections 412/412, joints2d 0.024 px, joints3d 0.024 mm, rotations 0.016° |
| Contact head | PyTorch fp32, 16 crops | logits within 0.010; contact decision identical on 6400/6400 points, 16/16 hands |

## Setup

Needs Apple silicon, macOS 15 / iOS 18, Xcode 16. The fast path uses Metal 4 tensor ops (M5 / A19 class GPUs,
OS 26.2+); other GPUs fall back to `simdgroup_matrix` automatically.

1. Install [FAH](https://github.com/noahcao/FAH) with its weights, the SAM 3D Body config and MANO (see its README),
   and build its Core ML detector (`fah-build-coreml --backbone s`).
2. Export once, from FAH's checkout:

   ```bash
   .venv/bin/python /path/to/fah-metal/tools/export.py --out /path/to/fah-metal/models
   ```

   Writes the network weights (77 MB with the contact head), the detector (54 MB) and MANO data (2.7 MB). They are
   derived from the FAH weights and MANO: don't commit or redistribute them.

## Use

```bash
swift run -c release fah-viewer --mesh             # camera (incl. Continuity Camera); --video clip.mp4 for a file
swift run -c release fah-check speed --models models --video clip.mp4
```

Viewer keys: `m` mesh, `c` contact (red where the hand touches), `r` rotate. Switching them off also stops computing
them; the overlay shows per-stage timings.

```swift
import FAHMetal

let fah = try FAHPipeline(modelDirectory: modelsURL)
let hands = try fah.process(pixelBuffer: buffer, intrinsics: K, rotation: .none)  // BGRA or 420f, any rotation
for h in hands {
    h.joints2d; h.joints3d             // 21 keypoints: pixels / camera frame (m)
    h.meshVertices                     // 778-vertex MANO mesh (faces: fah.meshFaces)
    h.rootRotation; h.localRotations   // MANO pose
    h.contact; h.contactFingers; h.contactScore   // per vertex / region / hand, probability
}
```

`FAHOptions`: `mesh`, `contact`, `meshFitIterations` (default 1), `detectorInterval` (default 10), `maxHands`.

## How it works

- **Pose network**: ~140 Metal dispatches in one command buffer. Matrix products on Metal 4 tensor ops, with LayerNorm,
  GELU, RoPE, residuals and pixel shuffles fused into their epilogues; the keypoint heatmap as split-KV attention (the
  461 × 6400 map is never stored); fp16 storage, fp32 accumulation.
- **Contact head**: FAH's branch of the last 4 encoder blocks and its 4-layer transformer decoder, on the same command
  buffer; 32-wide-head attention on simdgroup matrices with key / value tiles shared in threadgroup memory.
- **Pipeline**: FAH's detector (Core ML) every 10th frame, GPU crops, decoding and per-part Procrustes on the CPU.
- **Mesh**: FAH's MANO fit (Gauss-Newton on pose, shape, rotation and translation to the dense surface and joints,
  Huber at 1 cm; the hand's shape carries over between frames), then a PnP onto the 2D keypoints. ~1 ms.
- **Check it**: `tools/check.py pose` and `tools/check.py pipeline` compare against Python FAH.

## Licences

Code: Apache-2.0, like FAH's code ([LICENSE](LICENSE), [NOTICE](NOTICE)). Parts of `pose.metal` adapt
[MLX](https://github.com/ml-explore/mlx) (MIT, © Apple). The exported files fall under FAH's
[WEIGHTS_LICENSE](https://github.com/noahcao/FAH/blob/main/WEIGHTS_LICENSE) (research, non-commercial) and the
[MANO licence](https://mano.is.tue.mpg.de/license.html); none are in this repo.
