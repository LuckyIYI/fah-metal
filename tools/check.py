#!/usr/bin/env python3
"""Check fah-metal against Python FAH.

    pipeline   Python FAH-S (image mode, Core ML path, mesh="none", no fingertip step) on frames of a video, then
               FAHPipeline (`fah-check pipeline`) on the same pixels; compares joints, surface, rotations, scores.
    pose       The pose network alone: PyTorch fp32 (CPU) on crops from a video, then FAHPoseNetwork (`fah-check pose`).

Run in FAH's environment, from FAH's checkout, after `swift build -c release` in this repository and
tools/export.py. The pipeline reference uses FAH's Core ML pose network and detector, so run `fah-build-coreml
--backbone s` first.

    cd FAH
    .venv/bin/python /path/to/fah-metal/tools/check.py pipeline --models /path/to/fah-metal/models --video clip.mp4 --out /tmp/fahcheck
    .venv/bin/python /path/to/fah-metal/tools/check.py pose --models /path/to/fah-metal/models --video clip.mp4 --out /tmp/fahpose

`pipeline` saves the frames as PNG and the Python results as python.json in --out once, and reuses them on later runs
(--redo to recompute). Extra Swift runs: --rotate 90 (the frames turned into a landscape "sensor" image, as from a
phone, and turned back by the pipeline), --yuv (420f input, as from a camera; Python still sees the BGR frame, so this
adds the chroma subsampling to the difference), --det-every N (tracking between detector runs; Python tracks too).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FAH_CHECK = os.path.join(HERE, "..", ".build", "release", "fah-check")
MIRROR = np.diag([-1.0, 1.0, 1.0])


def fah_check(*args):
    exe = os.path.normpath(FAH_CHECK)
    if not os.path.exists(exe):
        sys.exit(f"{exe} not found: run `swift build -c release` in {os.path.dirname(HERE)}")
    subprocess.run([exe, *map(str, args)], check=True)


def fah_paths(a):
    from fah.api import DEFAULT_SAM3D, DEFAULT_WEIGHTS, FAH_HOME

    a.weights = os.path.expanduser(a.weights or DEFAULT_WEIGHTS)
    a.sam3d = os.path.expanduser(a.sam3d or DEFAULT_SAM3D)
    a.mano = os.path.expanduser(a.mano or os.environ.get("MANO_DIR", os.path.join(FAH_HOME, "mano")))


# ------------------------------------------------------------------------------------------------------------------
# pipeline
# ------------------------------------------------------------------------------------------------------------------
def aa_to_mat(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def python_reference(a, out):
    """FAH-S on the frames: per hand joints, the dense surface and the MANO rotations of fah.mano_procrustes (each hand
    in its own convention: left = M R M, M = diag(-1, 1, 1), as FAHPipeline returns them)."""
    import cv2

    import fah
    from fah.mano_procrustes import ManoProcrustes

    kw = {"det_every": a.det_every} if a.det_every > 1 else {}
    model = fah.load("fah-s", mode="image", coreml=True, device="mps", weights=a.weights, sam3d=a.sam3d, mano_dir=a.mano,
                     mesh="none", tips=False, **kw)
    if model.method.est.student_coreml is None:
        sys.exit(f"FAH's Core ML pose network is missing in {a.weights}/coreml: run `fah-build-coreml --backbone s`")
    ppm = ManoProcrustes(a.mano)
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start, stop, step = ([int(x) if x else None for x in a.frames.split(":")] + [None, None])[:3]
    want = set(range(start or 0, min(stop or n, n), step or 1))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fx = a.fx or 0.5 * W / np.tan(np.radians(a.hfov) / 2)
    K = np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1]], np.float32)
    frames, times = [], []
    for idx in range(max(want) + 1):
        ok, img = cap.read()
        if not ok:
            break
        if idx not in want:
            continue
        cv2.imwrite(os.path.join(out, f"frame_{idx:05d}.png"), img)
        t0 = time.perf_counter()
        hands = model(img, K)
        times.append((time.perf_counter() - t0) * 1e3)
        res = []
        for h in hands:
            right = bool(h["is_right"])
            M = np.eye(3) if right else MIRROR
            R, s = ppm.rotations(np.asarray(h["verts3d_dense"], np.float64) @ M)
            R0, pose = ppm.local_pose(R)
            res.append(dict(
                is_right=right, score=float(h["score"]), bbox=np.asarray(h["bbox"], float).tolist(),
                joints2d=np.asarray(h["joints2d"], float).tolist(), joints3d=np.asarray(h["joints3d"], float).tolist(),
                verts3d_dense=np.asarray(h["verts3d_dense"], float).tolist(), root_rotation=(M @ R0 @ M).tolist(),
                local_rotations=[(M @ aa_to_mat(v) @ M).tolist() for v in pose.reshape(15, 3)], scale=float(s)))
        frames.append(dict(index=idx, hands=res))
    meta = dict(video=a.video, width=W, height=H, K=K.tolist(), det_every=a.det_every, frames=frames,
                median_ms=float(np.median(times[5:] or times)))
    with open(os.path.join(out, "python.json"), "w") as f:
        json.dump(meta, f)
    print(f"python: {len(frames)} frames, {sum(len(f['hands']) for f in frames)} hands, median {meta['median_ms']:.1f} ms/frame")


def to_upright(h, k, Ws, Hs):
    """A Swift hand of a run on the image turned k quarter turns counter-clockwise -> the upright image's frame."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])  # X_upright = Rz X_sensor

    def p2(p):
        x, y = p[..., 0], p[..., 1]
        return [p, np.stack([Hs - 1 - y, x], -1), np.stack([Ws - 1 - x, Hs - 1 - y], -1), np.stack([y, Ws - 1 - x], -1)][k]

    o = dict(h)
    o["joints2d"] = p2(np.asarray(h["joints2d"]))
    o["joints3d"] = np.asarray(h["joints3d"]) @ Rz.T
    o["verts3d_dense"] = np.asarray(h["verts3d_dense"]) @ Rz.T
    o["root_rotation"] = Rz @ np.asarray(h["root_rotation"])
    return o


def angle(R1, R2):
    R = np.asarray(R1).T @ np.asarray(R2)
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def compare(py, sw):
    """Per matched hand (same side, nearest wrist): joints2d px, joints3d / surface mm, rotations deg, score, scale."""
    k = int(round(sw["rotation_deg"] / 90)) % 4
    W, H = py["width"], py["height"]
    Ws, Hs = (H, W) if k % 2 else (W, H)
    swf = {f["index"]: f for f in sw["frames"]}
    e = {n: [] for n in ("j2", "j3", "dense", "root", "local", "score", "scale")}
    frames = same = py_only = sw_only = 0
    for f in py["frames"]:
        if f["index"] not in swf:
            continue
        frames += 1
        ph = f["hands"]
        sh = [to_upright(h, k, Ws, Hs) for h in swf[f["index"]]["hands"]]
        same += len(ph) == len(sh) and sorted(h["is_right"] for h in ph) == sorted(h["is_right"] for h in sh)
        used = set()
        for h in ph:
            cand = [(np.linalg.norm(np.subtract(h["joints2d"][0], g["joints2d"][0])), i) for i, g in enumerate(sh)
                    if i not in used and g["is_right"] == h["is_right"]]
            if not cand:
                py_only += 1
                continue
            g = sh[min(cand)[1]]
            used.add(min(cand)[1])
            e["j2"].append(np.linalg.norm(np.asarray(h["joints2d"]) - g["joints2d"], axis=-1))
            e["j3"].append(np.linalg.norm(np.asarray(h["joints3d"]) - g["joints3d"], axis=-1) * 1e3)
            e["dense"].append(np.linalg.norm(np.asarray(h["verts3d_dense"]) - g["verts3d_dense"], axis=-1) * 1e3)
            e["root"].append(angle(h["root_rotation"], g["root_rotation"]))
            e["local"].append([angle(x, y) for x, y in zip(h["local_rotations"], g["local_rotations"])])
            e["score"].append(abs(h["score"] - g["score"]))
            e["scale"].append(abs(h["scale"] - g["scale"]))
        sw_only += len(sh) - len(used)

    def st(x):
        x = np.concatenate([np.ravel(v) for v in x]) if x else np.zeros(0)
        return f"median {np.median(x):.3f}  p95 {np.percentile(x, 95):.3f}  max {x.max():.3f}" if len(x) else "n/a"

    print(f"rotation {k * 90} deg, yuv {sw['yuv']}, det_every {sw['det_every']}, "
          f"pose network on {'tensor ops' if sw.get('tensor_ops') else 'simdgroup matrices'}")
    print(f"  frames {frames}: same hands and sides in {same}; matched hands {len(e['j2'])}, python only {py_only}, "
          f"swift only {sw_only}")
    for name, key in (("joints2d px", "j2"), ("joints3d mm", "j3"), ("dense surface mm", "dense"), ("root rotation deg", "root"),
                      ("local rotations deg", "local"), ("score |diff|", "score"), ("hand scale |diff|", "scale")):
        print(f"  {name:20s} {st(e[key])}")
    for name, t in sw["timings"].items():
        print(f"  swift timing, {name.replace('_', ' ')}: median {t['total']:.2f} ms | detector {t['detector']:.2f}, "
              f"crop {t['crop']:.2f}, pose {t['pose']:.2f} ({t['crops']:.0f} crops), decode {t['decode']:.2f}")


def pipeline(a):
    fah_paths(a)
    os.makedirs(a.out, exist_ok=True)
    ref = os.path.join(a.out, "python.json")
    if a.redo or not os.path.exists(ref):
        python_reference(a, a.out)
    py = json.load(open(ref))
    if py["det_every"] != a.det_every:
        sys.exit(f"{ref} was made with --det-every {py['det_every']}: use another --out or --redo")
    print(f"python: median {py['median_ms']:.1f} ms/frame (FAH-S, Core ML, mesh none)")
    for rot in a.rotate:
        out = os.path.join(a.out, f"swift_r{rot}{'_yuv' if a.yuv else ''}.json")
        args = ["pipeline", "--frames", a.out, "--models", a.models, "--out", out, "--rotate", rot, "--det-every", a.det_every,
                "--backend", a.backend]
        fah_check(*args, *(["--yuv"] if a.yuv else []))
        compare(py, json.load(open(out)))


# ------------------------------------------------------------------------------------------------------------------
# pose network
# ------------------------------------------------------------------------------------------------------------------
def pose(a):
    """Crops: random 600-px windows of the video's frames (every 15th) resized to 320, with plausible ray / cond inputs."""
    import cv2
    import torch

    import fah
    from fah.api import REPO

    fah_paths(a)
    sys.path.insert(0, REPO)
    from scripts.build_coreml import pose_net

    rng = np.random.default_rng(0)
    cap = cv2.VideoCapture(a.video)
    imgs = []
    S = 320
    while len(imgs) < a.n:
        for _ in range(15):
            ok, fr = cap.read()
        if not ok:
            break
        side = min(600, *fr.shape[:2])
        y, x = rng.integers(0, fr.shape[0] - side + 1), rng.integers(0, fr.shape[1] - side + 1)
        crop = cv2.resize(fr[y:y + side, x:x + side], (S, S))[:, :, ::-1]
        imgs.append(crop.transpose(2, 0, 1).astype(np.float32) / 255)
    n = len(imgs)
    img = np.stack(imgs)
    ray = np.tile(np.array([[600 / S / 1500, -0.2, 600 / S / 1500, -0.1]], np.float32), (n, 1)) + \
        rng.normal(0, 0.02, (n, 4)).astype(np.float32)
    cond = rng.normal(0, 0.3, (n, 3)).astype(np.float32)

    contact = os.path.exists(os.path.join(a.weights, "fah-s", "contact.pt"))
    model = fah.load("fah-s", mode="image", device="cpu", weights=a.weights, sam3d=a.sam3d, mano_dir=a.mano, mesh="none",
                     tips=False, contact=contact)
    net = pose_net(model.method.est, True, True, False)  # FAH's Core ML export graph (heads, folded adapter, no parts), fp32
    kh = net.kh  # without the part head the graph runs a copy of the keypoint head
    if contact:  # the contact branch reads the encoder's hidden state before its last blocks: hand it to that copy
        def grab(mod, args, kwargs):
            kh._br_in = (args[0], kwargs)
        net.enc.vit.blocks[kh.contact_branch_block].register_forward_pre_hook(grab, with_kwargs=True)
    with torch.no_grad():
        outs = net(torch.from_numpy(img), torch.from_numpy(ray), torch.from_numpy(cond))
    tensors = dict(img=img, ray=ray, cond=cond, **{k: v.float().numpy() for k, v in zip(net.outs, outs)})
    if contact:  # point (n, 400, K) and hand (n, K) logits, calibration offset included
        tensors.update(contact=kh.last_contact_k.float().numpy(), contact_hand=kh.last_contact_hand_k.float().numpy())
    os.makedirs(a.out, exist_ok=True)
    meta, off = {}, 0
    with open(os.path.join(a.out, "ref.bin"), "wb") as f:
        for k, v in tensors.items():
            v = np.ascontiguousarray(v, np.float32)
            meta[k] = dict(offset=off, shape=list(v.shape))
            f.write(v.tobytes())
            off += v.nbytes
    with open(os.path.join(a.out, "ref.json"), "w") as f:
        json.dump(dict(n=n, tensors=meta), f)
    print(f"PyTorch fp32 reference: {n} crops -> {a.out}/ref.bin", flush=True)
    fah_check("pose", "--models", a.models, "--ref", a.out, "--backend", a.backend, "--iters", a.iters)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pipeline", "pose"):
        p = sub.add_parser(name)
        p.add_argument("--models", required=True, help="tools/export.py's output folder")
        p.add_argument("--video", required=True)
        p.add_argument("--out", required=True, help="working folder (frames, references, results)")
        p.add_argument("--backend", default="auto", choices=["auto", "tensor", "simdgroup"], help="pose network kernels")
        p.add_argument("--weights", help="FAH weights (default: FAH's)")
        p.add_argument("--sam3d", help="SAM 3D Body config (default: FAH's)")
        p.add_argument("--mano", help="folder with MANO_RIGHT.pkl (default: FAH's)")
    p = sub.choices["pipeline"]
    p.add_argument("--frames", default="0:", help="start:stop[:step] (default: all)")
    p.add_argument("--fx", type=float, default=0, help="focal length in px (default: from --hfov)")
    p.add_argument("--hfov", type=float, default=70, help="horizontal field of view, degrees, when --fx is not given")
    p.add_argument("--det-every", type=int, default=1, help="detector schedule (1: every frame)")
    p.add_argument("--rotate", nargs="+", default=["0"], help="Swift runs on the frames turned by these angles")
    p.add_argument("--yuv", action="store_true", help="Swift sees 420f buffers")
    p.add_argument("--redo", action="store_true", help="recompute the Python reference")
    p = sub.choices["pose"]
    p.add_argument("--n", type=int, default=16, help="number of crops")
    p.add_argument("--iters", type=int, default=200, help="timed runs per batch size")
    a = ap.parse_args()
    a.models, a.out, a.video = (os.path.abspath(os.path.expanduser(x)) for x in (a.models, a.out, a.video))
    pipeline(a) if a.cmd == "pipeline" else pose(a)


if __name__ == "__main__":
    main()
