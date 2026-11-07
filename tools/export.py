#!/usr/bin/env python3
"""Export everything fah-metal needs from a local FAH install into one folder.

    fah-s-pose.bin / fah-s-pose.json   the FAH-S pose network's weights in the layout of the Metal kernels, with FAH's
                                       hand-object contact head when <weights>/fah-s/contact.pt exists
    detector-640.mlpackage             FAH's Core ML hand detector (copied from fah-build-coreml's output, or built)
    fah_mano.bin                       MANO-derived data: the per-part Procrustes template, FAH-S's 400 -> 778 surface
                                       map and the mesh faces (for drawing)

Run it once, in FAH's environment, from FAH's checkout (https://github.com/noahcao/FAH):

    cd FAH
    .venv/bin/python /path/to/fah-metal/tools/export.py --out /path/to/fah-metal/models

It reads the FAH weights, the SAM 3D Body config and MANO from FAH's usual places ($FAH_WEIGHTS, $FAH_SAM3D,
$MANO_DIR, else under $FAH_HOME, default ~/.cache/fah); pass --weights / --sam3d / --mano to override. The outputs
are derived from the FAH weights (FAH's WEIGHTS_LICENSE) and from MANO (the MANO licence): do not commit or
redistribute them.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys

import numpy as np

POSE_NAME = "fah-s-pose"
DETECTOR_NAME = "detector-640"
MANO_NAME = "fah_mano.bin"


# ------------------------------------------------------------------------------------------------------------------
# Pose network weights
# ------------------------------------------------------------------------------------------------------------------
class TensorFile:
    """A flat blob of tensors (offsets 256-byte aligned) plus a JSON index: name -> offset, shape, dtype."""

    def __init__(self):
        self.blob, self.meta, self.off = [], {}, 0

    def add(self, name, arr, dtype):
        assert name not in self.meta, name
        arr = np.ascontiguousarray(np.asarray(arr, np.float32).astype(np.float32 if dtype == "f32" else np.float16))
        pad = (-self.off) % 256
        self.blob.append(b"\0" * pad)
        self.off += pad
        self.meta[name] = dict(offset=self.off, shape=list(arr.shape), dtype=dtype)
        self.blob.append(arr.tobytes())
        self.off += arr.nbytes

    def save(self, stem, extra):
        with open(stem + ".bin", "wb") as f:
            f.writelines(self.blob)
        with open(stem + ".json", "w") as f:
            json.dump(dict(tensors=self.meta, **extra), f, indent=1)


def f32(t):
    """A torch tensor (any device / dtype) as a float32 array."""
    return t.detach().float().cpu().numpy()


def sincos_2d(c, h, w):
    """(h*w, c) fixed 2D sin-cos positions (FAH keypoint_head.sincos_2d, as token rows y*w + x)."""
    k = np.arange(c // 4)
    fr = 1.0 / (100.0 ** (k / max(c // 4 - 1, 1)))
    ys, xs = (np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w
    ex = np.broadcast_to((xs[None] * fr[:, None] * 2 * np.pi)[:, None, :], (c // 4, h, w))
    ey = np.broadcast_to((ys[None] * fr[:, None] * 2 * np.pi)[:, :, None], (c // 4, h, w))
    return np.concatenate([np.sin(ex), np.cos(ex), np.sin(ey), np.cos(ey)], 0).reshape(c, h * w).T


def export_contact(w, path):
    """FAH's hand-object contact head (contact.pt, trained on the frozen network) into the pose network's tensor file:
    the branch (copies of the encoder's last blocks, same layout as the encoder's), the pooled heatmap-feature
    memory, the per-point inputs, the transformer decoder layers and the point / hand outputs. Returns its config."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.split("keypoint_head.", 1)[1]: v.double().numpy() for k, v in ck["trainable"].items()}
    T = lambda k: sd[k].T  # noqa: E731  torch Linear (N, K) -> GEMM right operand (K, N)
    nb = len({k.split(".")[2] for k in sd if k.startswith("contact_branch.blocks.")})
    for i in range(nb):
        b = f"contact_branch.blocks.{i}."
        for k, src in (("n1w", "norm1.weight"), ("n1b", "norm1.bias"), ("n2w", "norm2.weight"), ("n2b", "norm2.bias"),
                       ("projb", "attn.proj.bias"), ("g1", "gamma_1"), ("fc1b", "mlp.fc1.bias"), ("fc2b", "mlp.fc2.bias"),
                       ("g2", "gamma_2")):
            w.add(f"cb{i}.{k}", sd[b + src], "f32")
        for k, src in (("qkv", "attn.qkv.weight"), ("proj", "attn.proj.weight"), ("fc1", "mlp.fc1.weight"),
                       ("fc2", "mlp.fc2.weight")):
            w.add(f"cb{i}.{k}", T(b + src), "f16")
    w.add("cb.norm_w", sd["contact_branch.norm.weight"], "f32")
    w.add("cb.norm_b", sd["contact_branch.norm.bias"], "f32")
    w.add("cb.proj_lnw", sd["contact_branch.proj.0.weight"], "f32")
    w.add("cb.proj_lnb", sd["contact_branch.proj.0.bias"], "f32")
    w.add("cb.proj_w", T("contact_branch.proj.1.weight"), "f16")  # (384, 256)
    w.add("cb.proj_b", sd["contact_branch.proj.1.bias"], "f32")
    cd = sd["contact_tok.bias"].shape[0]
    w.add("c.pos_mem", sincos_2d(cd, 20, 20), "f32")  # (400, 256)
    w.add("c.img_w", sd["contact_img.0.weight"][:, :, 0, 0].T, "f16")  # (64, 256)
    w.add("c.img_b", sd["contact_img.0.bias"], "f32")
    w.add("c.img_lnw", sd["contact_img.1.weight"], "f32")
    w.add("c.img_lnb", sd["contact_img.1.bias"], "f32")
    w.add("c.tok_w", T("contact_tok.weight"), "f16")  # (1024, 256)
    w.add("c.tok_b", sd["contact_tok.bias"], "f32")
    w.add("c.pos", sd["contact_pos"], "f32")  # (400, 256)
    w.add("c.in_lnw", sd["contact_in.0.weight"], "f32")  # 132 = pooled 64 | query 64 | xyz * 10 | vis
    w.add("c.in_lnb", sd["contact_in.0.bias"], "f32")
    w.add("c.in_w1", T("contact_in.1.weight"), "f16")  # (132, 256)
    w.add("c.in_b1", sd["contact_in.1.bias"], "f32")
    w.add("c.in_w2", T("contact_in.3.weight"), "f16")  # (256, 256)
    w.add("c.in_b2", sd["contact_in.3.bias"], "f32")
    nl = len({k.split(".")[1] for k in sd if k.startswith("contact_mix.")})
    kv_w, kv_b = [], []
    for L in range(nl):
        m = f"contact_mix.{L}."
        for j in (1, 2, 3):
            w.add(f"c{L}.n{j}w", sd[f"{m}norm{j}.weight"], "f32")
            w.add(f"c{L}.n{j}b", sd[f"{m}norm{j}.bias"], "f32")
        w.add(f"c{L}.sa_w", sd[m + "self_attn.in_proj_weight"].T, "f16")  # (256, 768): q | k | v
        w.add(f"c{L}.sa_b", sd[m + "self_attn.in_proj_bias"], "f32")
        w.add(f"c{L}.sa_ow", T(m + "self_attn.out_proj.weight"), "f16")
        w.add(f"c{L}.sa_ob", sd[m + "self_attn.out_proj.bias"], "f32")
        ca_w, ca_b = sd[m + "multihead_attn.in_proj_weight"], sd[m + "multihead_attn.in_proj_bias"]
        w.add(f"c{L}.ca_qw", ca_w[:cd].T, "f16")
        w.add(f"c{L}.ca_qb", ca_b[:cd], "f32")
        kv_w.append(ca_w[cd:].T)  # every layer's memory keys / values in one GEMM: (256, layers * 512)
        kv_b.append(ca_b[cd:])
        w.add(f"c{L}.ca_ow", T(m + "multihead_attn.out_proj.weight"), "f16")
        w.add(f"c{L}.ca_ob", sd[m + "multihead_attn.out_proj.bias"], "f32")
        w.add(f"c{L}.l1w", T(m + "linear1.weight"), "f16")  # (256, 512), ReLU
        w.add(f"c{L}.l1b", sd[m + "linear1.bias"], "f32")
        w.add(f"c{L}.l2w", T(m + "linear2.weight"), "f16")
        w.add(f"c{L}.l2b", sd[m + "linear2.bias"], "f32")
    w.add("c.kv_w", np.concatenate(kv_w, 1), "f16")
    w.add("c.kv_b", np.concatenate(kv_b), "f32")
    w.add("c.ones", np.ones(cd), "f32")  # unit layer scale for the residual + LayerNorm GEMM epilogue
    w.add("c.out_lnw", sd["contact_out.0.weight"], "f32")
    w.add("c.out_lnb", sd["contact_out.0.bias"], "f32")
    w.add("c.out_w", T("contact_out.1.weight"), "f32")  # (256, K tolerances)
    w.add("c.out_b", sd["contact_out.1.bias"] - sd["contact_bias"], "f32")  # FAH's calibration offset folded in
    w.add("c.hand_lnw", sd["contact_hand.0.weight"], "f32")
    w.add("c.hand_lnb", sd["contact_hand.0.bias"], "f32")
    w.add("c.hand_w1", T("contact_hand.1.weight"), "f32")  # (512, 64)
    w.add("c.hand_b1", sd["contact_hand.1.bias"], "f32")
    w.add("c.hand_w2", T("contact_hand.3.weight"), "f32")  # (64, K)
    w.add("c.hand_b2", sd["contact_hand.3.bias"], "f32")
    dists = [float(x) for x in sd["contact_dists"]]
    return dict(branch_blocks=nb, layers=nl, dim=cd, heads=max(4, cd // 32), points=int(sd["contact_pos"].shape[0]),
                dists_mm=dists, main=dists.index(10.0))


def export_pose(est, stem, contact=None):
    """FAH-S's decoder-free pose network (as FAH's Core ML export, scripts/build_coreml.py PoseNet, without the part
    head) in the layout fah-metal's kernels read. Matmul weights are fp16 (K, N) row-major; vectors, LayerNorm
    parameters and RoPE tables fp32. Linear layers that follow each other without a non-linearity are folded in fp64."""
    import torch
    import torch.nn.functional as F

    enc, m = est.student, est.model
    vit, kh = enc.vit, m.keypoint_head
    assert np.allclose(f32(enc.mean), f32(m.image_mean)) and np.allclose(f32(enc.std), f32(m.image_std))
    assert kh.contact_on is False and getattr(kh, "local_r", 0) == 0 and len(kh.up) == 5, "unexpected FAH-S keypoint head"
    assert kh.vis is not None and kh.depth is not None and getattr(m, "verify_head", None) is not None
    S, g, P = enc.input_size, enc.grid, enc.num_prefix
    w = TensorFile()
    T = lambda lin: f32(lin.weight).T  # noqa: E731  torch Linear (out, in) -> (in, out)
    conv = lambda c: f32(c.weight).transpose(0, 2, 3, 1).reshape(c.weight.shape[0], -1).T  # noqa: E731  -> (kh kw I, O)

    # ---- ViT: patch embedding, prefix tokens, RoPE, blocks
    pw = f32(vit.patch_embed.proj.weight)  # (D, 3, p, p) -> rows (ky, kx, c)
    D, p = pw.shape[0], pw.shape[-1]
    w.add("patch_w", pw.transpose(2, 3, 1, 0).reshape(-1, D), "f16")
    w.add("patch_b", f32(vit.patch_embed.proj.bias), "f32")
    prefix = f32(torch.cat([vit.cls_token, vit.reg_token], 1))[0]  # rows 0..4 of the residual stream
    w.add("prefix_x", prefix, "f32")
    n1 = vit.blocks[0].norm1  # block 0's LN1 of the prefix rows is input independent
    mu, var = prefix.mean(-1, keepdims=True), prefix.var(-1, keepdims=True)
    w.add("prefix_y", (prefix - mu) / np.sqrt(var + n1.eps) * f32(n1.weight) + f32(n1.bias), "f16")
    rope = {}  # the blocks' RoPE keyword argument is fixed for the fixed patch grid
    hook = vit.blocks[0].register_forward_pre_hook(lambda mod, a, kw: rope.update(kw), with_kwargs=True)
    with torch.no_grad():
        enc(torch.zeros(1, 3, S, S, device=next(enc.parameters()).device))
    hook.remove()
    assert getattr(vit.blocks[0].attn, "rotate_half", False)
    sin, cos = np.split(f32(rope["rope"]), 2, -1)
    w.add("rope_cos", cos.reshape(g * g, -1), "f32")
    w.add("rope_sin", sin.reshape(g * g, -1), "f32")
    for i, b in enumerate(vit.blocks):
        a = b.attn
        assert a.q_bias is None and isinstance(a.q_norm, torch.nn.Identity) and isinstance(a.norm, torch.nn.Identity)
        assert b.gamma_1 is not None and b.gamma_2 is not None
        for k, t in (("n1w", b.norm1.weight), ("n1b", b.norm1.bias), ("n2w", b.norm2.weight), ("n2b", b.norm2.bias),
                     ("projb", a.proj.bias), ("g1", b.gamma_1), ("fc1b", b.mlp.fc1.bias), ("fc2b", b.mlp.fc2.bias),
                     ("g2", b.gamma_2)):
            w.add(f"b{i}.{k}", f32(t), "f32")
        for k, lin in (("qkv", a.qkv), ("proj", a.proj), ("fc1", b.mlp.fc1), ("fc2", b.mlp.fc2)):
            w.add(f"b{i}.{k}", T(lin), "f16")
    w.add("norm_w", f32(vit.norm.weight), "f32")
    w.add("norm_b", f32(vit.norm.bias), "f32")

    # ---- adapter, folded into the ray embedding's 1x1 conv together with the no-mask prompt embedding
    ad, ce = enc.adapter, m.ray_cond_emb_hand
    w.add("ad_lnw", f32(ad[0].weight), "f32")
    w.add("ad_lnb", f32(ad[0].bias), "f32")
    w.add("ad_w", T(ad[1]), "f16")
    w.add("ad_b", f32(ad[1].bias), "f32")
    with torch.no_grad():
        W = ce.conv.weight[:, :, 0, 0].double().cpu()
        Wi, Wr = W[:, :enc.out_dim], W[:, enc.out_dim:]
        if m.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
            no_mask = m.prompt_encoder.no_mask_embed.weight.detach().reshape(-1).double().cpu()
        else:
            no_mask = torch.zeros(W.shape[0], dtype=torch.float64)
        w_fold = (Wi @ ad[3].weight.double().cpu()).T.float().numpy()
        b_fold = (Wi @ (ad[3].bias.double().cpu() + no_mask)).float().numpy()
        w_ray = Wr.T.float().numpy()
    # one GEMM over [adapter hidden | ray embedding | zero padding to a multiple of 128]
    kin = w_fold.shape[0] + w_ray.shape[0]
    w_cat = np.zeros((kin + (-kin) % 128, w_fold.shape[1]), np.float32)
    w_cat[:w_fold.shape[0]], w_cat[w_fold.shape[0]:kin] = w_fold, w_ray
    w.add("w_cat", w_cat, "f16")
    w.add("b_fold", b_fold, "f32")
    w.add("ce_w", f32(ce.norm.weight), "f32")
    w.add("ce_b", f32(ce.norm.bias), "f32")
    # the ray grid: the antialiased 1/patch downsample of the pixel grid (linear, so computed once)
    vv, uu = torch.meshgrid(torch.arange(S, dtype=torch.float32), torch.arange(S, dtype=torch.float32), indexing="ij")
    uv_ds = F.interpolate(torch.stack([uu, vv])[None], scale_factor=(1 / ce.patch_size,) * 2, mode="bilinear",
                          align_corners=False, antialias=True)[0]
    w.add("uv_ds", uv_ds.permute(1, 2, 0).reshape(g * g, 2).numpy(), "f32")
    cam = ce.camera  # Fourier camera embedding
    w.add("freq", np.stack([np.linspace(1.0, r / 2, cam.num_bands) for r in cam.max_resolution]).reshape(-1), "f32")

    # ---- pooled pose token and the MLPs on it
    pool_w = T(kh.pool[1])  # (1283, 1024), transposed and K zero-padded for the GEMV
    pool_wt = np.zeros((pool_w.shape[1], pool_w.shape[0] + (-pool_w.shape[0]) % 32), np.float32)
    pool_wt[:, :pool_w.shape[0]] = pool_w.T
    w.add("pool_lnw", f32(kh.pool[0].weight), "f32")
    w.add("pool_lnb", f32(kh.pool[0].bias), "f32")
    w.add("pool_wt", pool_wt, "f16")
    w.add("pool_b", f32(kh.pool[1].bias), "f32")
    vh = m.verify_head
    d = 2 * kh.dim
    x0 = f32(kh.xyz[0].weight)  # (256, pooled + query | pose)
    # first layers of the query, depth and verify MLPs and the pose part of the xyz MLP: one (1280, 1024) GEMV
    w.add("ph_wt", np.concatenate([f32(kh.query[0].weight), f32(kh.depth[1].weight), f32(vh.mlp[0].weight), x0[:, d:]]),
          "f16")
    w.add("ph_b", np.concatenate([f32(kh.query[0].bias), f32(kh.depth[1].bias), f32(vh.mlp[0].bias), f32(kh.xyz[0].bias)]),
          "f32")
    w.add("depth_lnw", f32(kh.depth[0].weight), "f32")
    w.add("depth_lnb", f32(kh.depth[0].bias), "f32")
    w.add("q1_wt", f32(kh.query[2].weight), "f16")
    w.add("q1_b", f32(kh.query[2].bias), "f32")
    w.add("d1_w", f32(kh.depth[3].weight).reshape(-1), "f32")
    w.add("d1_b", f32(kh.depth[3].bias), "f32")
    w.add("a1_wt", f32(vh.mlp[2].weight), "f32")
    w.add("a1_b", f32(vh.mlp[2].bias), "f32")
    w.add("point", f32(kh.point), "f32")
    w.add("qmlp_lnw", f32(kh.qmlp[0].weight), "f32")
    w.add("qmlp_lnb", f32(kh.qmlp[0].bias), "f32")
    w.add("qmlp_m0", T(kh.qmlp[1]), "f16")
    w.add("qmlp_m0b", f32(kh.qmlp[1].bias), "f32")
    w.add("qmlp_m1", T(kh.qmlp[3]), "f16")
    w.add("qmlp_m1b", f32(kh.qmlp[3].bias), "f32")
    w.add("xyz_a", x0[:, :d].T, "f16")
    w.add("xyz_w2t", f32(kh.xyz[2].weight), "f32")
    w.add("xyz_b2", f32(kh.xyz[2].bias), "f32")
    w.add("vis_lnw", f32(kh.vis[0].weight), "f32")
    w.add("vis_lnb", f32(kh.vis[0].bias), "f32")
    w.add("vis_w0", T(kh.vis[1]), "f16")
    w.add("vis_b0", f32(kh.vis[1].bias), "f32")
    w.add("vis_w1", f32(kh.vis[3].weight).reshape(-1), "f32")
    w.add("vis_b1", f32(kh.vis[3].bias), "f32")

    # ---- keypoint head convolutions as GEMMs; 2x2 stride-2 transposed convs as a GEMM + pixel shuffle
    def conv_t(c):
        wt = f32(c.weight)  # (I, O, 2, 2) -> (I, (di 2 + dj) O + o)
        return wt.transpose(0, 2, 3, 1).reshape(wt.shape[0], -1), np.tile(f32(c.bias), 4)

    w.add("ip_w", conv(kh.img_proj[0]), "f16")
    w.add("ip_b", f32(kh.img_proj[0].bias), "f32")
    w.add("ip_lnw", f32(kh.img_proj[1].weight), "f32")
    w.add("ip_lnb", f32(kh.img_proj[1].bias), "f32")
    for name, c in (("up0", kh.up[0]), ("up1", kh.up[3])):
        cw, cb = conv_t(c)
        w.add(f"{name}_w", cw, "f16")
        w.add(f"{name}_b", cb, "f32")
        if name == "up0":
            w.add("up0_lnw", f32(kh.up[1].weight), "f32")
            w.add("up0_lnb", f32(kh.up[1].bias), "f32")
    w.add("stem0_w", f32(kh.stem[0].weight).transpose(0, 2, 3, 1), "f32")  # (32, 3, 3, 3) direct conv
    w.add("stem0_b", f32(kh.stem[0].bias), "f32")
    w.add("stem1_w", conv(kh.stem[2]), "f16")
    w.add("stem1_b", f32(kh.stem[2].bias), "f32")
    fz = kh.fuse
    for name, c, ln in (("fuse0", fz[0], fz[1]), ("fuse1", fz[3], fz[4])):
        w.add(f"{name}_w", conv(c), "f16")
        w.add(f"{name}_b", f32(c.bias), "f32")
        w.add(f"{name}_lnw", f32(ln.weight), "f32")
        w.add(f"{name}_lnb", f32(ln.bias), "f32")
    w.add("fuse2_w", conv(fz[6]), "f16")
    w.add("fuse2_b", f32(fz[6].bias), "f32")
    w.add("mean", f32(enc.mean).reshape(-1), "f32")
    w.add("std", f32(enc.std).reshape(-1), "f32")

    eps = dict(ln1=[b.norm1.eps for b in vit.blocks], ln2=[b.norm2.eps for b in vit.blocks], norm=vit.norm.eps,
               ad=ad[0].eps, ce=ce.norm.eps, ip=kh.img_proj[1].eps, up0=kh.up[1].eps, fuse0=fz[1].eps, fuse1=fz[4].eps,
               pool=kh.pool[0].eps, depth=kh.depth[0].eps, qmlp=kh.qmlp[0].eps, vis=kh.vis[0].eps)
    cfg = dict(S=S, patch=p, grid=g, prefix=P, dim=D, heads=vit.blocks[0].attn.num_heads, blocks=len(vit.blocks),
               mlp=vit.blocks[0].mlp.fc1.weight.shape[0], kh_dim=kh.dim, points=kh.point.shape[0],
               temp=float(kh.log_temp.exp().item() / np.sqrt(kh.dim)), eps=eps)
    if contact:
        cfg["contact"] = export_contact(w, contact)
    w.save(stem, dict(config=cfg))
    print(f"{stem}.bin: {w.off / 1e6:.1f} MB, {len(w.meta)} tensors{' (with the contact head)' if contact else ''}")


# ------------------------------------------------------------------------------------------------------------------
# MANO data
# ------------------------------------------------------------------------------------------------------------------
def export_mano(mano_dir, checkpoint, path):
    """fah_mano.bin, little endian (read by Sources/FAHMetal/Mano.swift):

        magic "FAHMANO4", int32 nv (778), nparts (16), ndense (400), nbone (40), nfaces (1538)
        float64 v_template (nv, 3)   MANO right-hand template (flat hand, mean shape)
        int32   part (nv)            argmax of the skinning weights: the Procrustes part of each vertex
        int32   parents (nparts)     kinematic tree (root -1)
        int32   dense_vidx (ndense)  template vertex of each of the network's dense surface points
        float32 sub_R (nv, ndense)   dense points -> 778-vertex surface (FAH's weighted spectral fit,
                                     fah.estimator._kp_only_results)
        int32   faces (nfaces, 3)    MANO right-hand triangles (left hands: reverse the winding)
        float64 joints (nparts, 3)   MANO's rest joints of the template (J_regressor @ v_template)
        float32 weights (nv, nparts) MANO's skinning weights (the posed template mesh, ManoProcrustes.__call__)
        float32 shapedirs (10, nv * 3)        the first 10 shape directions (betas), per vertex coordinate
        float32 posedirs (135, nv * 3)        the pose-corrective directions of the 15 joints' (R - I) entries
        float32 joint_dirs (10, nparts * 3)   the rest joints' shape directions (J_regressor @ shapedirs)
        float32 J_regressor (nparts, nv)      joints of a mesh (the fitted mesh's 2D alignment)
    """
    import torch

    from fah.models.heads.spectral import template_basis

    with open(os.path.join(mano_dir, "MANO_RIGHT.pkl"), "rb") as f:
        m = pickle.load(f, encoding="latin1")
    dense = lambda x: np.asarray(x.todense() if hasattr(x, "todense") else x, np.float64)  # noqa: E731
    v = dense(m["v_template"])
    parents = np.asarray(m["kintree_table"][0], np.int64)
    parents[0] = -1
    W = dense(m["weights"])
    part = W.argmax(1)
    Jreg = dense(m["J_regressor"])
    J = Jreg @ v
    sd = np.asarray(m["shapedirs"], np.float64)[..., :10]  # (nv, 3, 10)
    pdirs = np.asarray(m["posedirs"], np.float64)  # (nv, 3, 135)
    shapedirs = sd.reshape(len(v) * 3, 10).T
    posedirs = pdirs.reshape(len(v) * 3, 135).T
    joint_dirs = np.einsum("jv,vcs->jcs", Jreg, sd).reshape(-1, 10).T
    faces = np.asarray(m["f"], np.int64)

    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["trainable"]
    nb, nd = (int(x) for x in sd["model.keypoint_head.dense"])
    vi = sd["model.keypoint_head.dense_vidx"].cpu().numpy().astype(np.int64)
    assert len(vi) == nd
    U, M = template_basis(mano_dir, min(256, nd // 2))  # as HandEstimator._kp_only_results
    U, mw = U.astype(np.float64), M.astype(np.float64)[vi]
    UW = U[vi].T * mw[None]
    sub_R = U @ np.linalg.solve(UW @ U[vi] + 1e-3 * np.eye(U.shape[1]), UW)

    with open(path, "wb") as f:
        f.write(b"FAHMANO4")
        f.write(struct.pack("<5i", len(v), len(parents), nd, nb, len(faces)))
        for arr, dt in ((v, "<f8"), (part, "<i4"), (parents, "<i4"), (vi, "<i4"), (sub_R, "<f4"),
                        (faces, "<i4"), (J, "<f8"), (W, "<f4"), (shapedirs, "<f4"), (posedirs, "<f4"),
                        (joint_dirs, "<f4"), (Jreg, "<f4")):
            f.write(np.ascontiguousarray(arr).astype(dt).tobytes())
    print(f"{path}: {os.path.getsize(path) / 1e6:.2f} MB (nv {len(v)}, dense {nd}, faces {len(faces)})")


# ------------------------------------------------------------------------------------------------------------------
# Detector
# ------------------------------------------------------------------------------------------------------------------
def export_detector(weights, out):
    """FAH's Core ML detector: fah-build-coreml's export when it exists, else built here (needs coremltools)."""
    from fah.api import REPO, coreml_paths, detector_weights
    from fah.build_coreml import DEFAULT_FRAMES

    dst = os.path.join(out, DETECTOR_NAME + ".mlpackage")
    src = coreml_paths("fah-s", weights)[1]
    if os.path.exists(dst):
        shutil.rmtree(dst)
    if os.path.exists(src):
        shutil.copytree(src, dst)
        print(f"{dst}: copied from {src}")
        return
    print(f"{src} not found: building the detector with FAH's scripts/build_coreml.py (a few minutes)")
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "build_coreml.py"), "det", "--weights",
                    detector_weights(weights), "--imgsz", "640", "--frames", DEFAULT_FRAMES, "--out", dst],
                   check=True, cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO))
    print(f"{dst}: built")


def main():
    from fah.api import DEFAULT_SAM3D, DEFAULT_WEIGHTS, FAH_HOME

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (the pipeline's model directory)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="FAH weights (default: %(default)s)")
    ap.add_argument("--sam3d", default=DEFAULT_SAM3D, help="SAM 3D Body config (default: %(default)s)")
    ap.add_argument("--mano", default=os.environ.get("MANO_DIR", os.path.join(FAH_HOME, "mano")),
                    help="folder with MANO_RIGHT.pkl (default: %(default)s)")
    ap.add_argument("--no-contact", dest="contact", action="store_false",
                    help="leave out the hand-object contact head (default: included when <weights>/fah-s/contact.pt exists)")
    a = ap.parse_args()
    a.weights, a.sam3d, a.mano, a.out = (os.path.abspath(os.path.expanduser(x)) for x in (a.weights, a.sam3d, a.mano, a.out))
    os.makedirs(a.out, exist_ok=True)

    import fah

    model = fah.load("fah-s", mode="image", device="cpu", weights=a.weights, sam3d=a.sam3d, mano_dir=a.mano, mesh="none",
                     tips=False)
    contact = os.path.join(a.weights, "fah-s", "contact.pt")
    export_pose(model.method.est, os.path.join(a.out, POSE_NAME), contact if a.contact and os.path.exists(contact) else None)
    export_mano(a.mano, os.path.join(a.weights, "fah-s", "model.pt"), os.path.join(a.out, MANO_NAME))
    export_detector(a.weights, a.out)


if __name__ == "__main__":
    main()
