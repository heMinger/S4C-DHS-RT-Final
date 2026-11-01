#!/usr/bin/env python
"""Fixed-batch forward/backward check for the right-keyframe semantic fix.

No optimizer step, no checkpoint write. Compares a fixed right-keyframe sample
(affected) and a left-keyframe sample (negative control) with the same seeds.
"""
import argparse, json, random, sys, hashlib
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--cp', required=True)
    ap.add_argument('--right-index', type=int, required=True)
    ap.add_argument('--left-index', type=int, required=True)
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    sys.path.insert(0, str(Path(args.repo) / 'scripts'))
    from datasets.data_util import make_datasets
    from models.bts.model.models_bts import BTSNet
    from models.bts.trainer import BTSWrapper
    from models.bts.model.loss import ReconstructionLoss
    from models.common.render import NeRFRenderer
    from utils.array_operations import to

    cfg = yaml.safe_load(open(args.config))
    data_cfg = dict(cfg['data'])
    data_cfg.update(
        data_path='/mnt/disk1/lmh/KITTI360',
        pose_path='/mnt/disk1/lmh/KITTI360/pose',
        data_segmentation_path='/mnt/disk1/lmh/KITTI360/mask2former_cityscapes_swinl_in21k_384_90k_ar_crop_192x640',
        data_preprocessed_path='/mnt/disk1/lmh/KITTI360/KITTI360_preprocessed/KITTI360_preprocessed',
        depth_da3_path='/mnt/disk1/lmh/KITTI360/data_2d_depth_da3_v3',
        split_path=str(Path(args.repo) / 'datasets/kitti_360/splits/sscbench'),
        segmentation_mode=cfg.get('segmentation_mode', 'panoptic_deeplab'),
    )
    train_ds, _ = make_datasets(data_cfg)
    train_ds.color_aug = False
    train_ds.random_fisheye_offset = False
    train_ds.additional_random_front_offset = False
    train_ds.return_depth = True
    train_ds.return_lidar_depth = False
    train_ds.return_segmentation = True

    device = torch.device(args.device)
    net = BTSNet(cfg['model_conf'])
    renderer = NeRFRenderer.from_conf(cfg['renderer'])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    model = BTSWrapper(renderer, cfg['model_conf'], False)
    cp = torch.load(args.cp, map_location='cpu', weights_only=False)
    sd = cp['model'] if isinstance(cp, dict) and 'model' in cp else cp
    inc = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.train()
    criterion = ReconstructionLoss(cfg['loss'], cfg['model_conf'].get('use_automasking', False))
    if tuple(net.ray_attention.proj_in.weight.shape) != (32, 39):
        raise RuntimeError('expected RT39')
    out = {
        'repo': str(Path(args.repo).resolve()),
        'loader_sha256': sha(Path(args.repo) / 'datasets/kitti_360/kitti_360_dataset.py'),
        'config': str(Path(args.config).resolve()),
        'cp_sha256': sha(args.cp),
        'seed': args.seed,
        'missing_keys': list(getattr(inc, 'missing_keys', [])),
        'unexpected_keys': list(getattr(inc, 'unexpected_keys', [])),
        'samples': {},
    }
    for label, index in [('right', args.right_index), ('left', args.left_index)]:
        random.seed(args.seed); np.random.seed(args.seed % (2**32)); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None
        seq, fid, is_right = train_ds._datapoints[index]
        loader = DataLoader(Subset(train_ds, [index]), batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=default_collate)
        data = next(iter(loader))
        keep = ['imgs', 'depths', 'confs', 'segs_gt', 'poses', 'projs']
        data = {k: data[k] for k in keep if k in data}
        data = to(data, device)
        model.zero_grad(set_to_none=True)
        rendered = model(data)
        loss, loss_dict = criterion(rendered)
        loss.backward()
        def gnorm(prefix):
            vals = [p.grad.detach().float().norm().item() ** 2 for n, p in model.named_parameters()
                    if p.grad is not None and n.startswith(prefix)]
            return float(np.sqrt(sum(vals))) if vals else 0.0
        g_all = [p.grad.detach().float() for p in model.parameters() if p.grad is not None]
        nonfinite = sum(int((~torch.isfinite(g)).sum().item()) for g in g_all)
        out['samples'][label] = {
            'dataset_index': int(index),
            'sequence': str(seq),
            'frame': int(fid),
            'is_right': bool(is_right),
            'loss': float(loss.detach().cpu()),
            'loss_dict': {k: float(v) for k, v in loss_dict.items()},
            'grad_norm_all': float(np.sqrt(sum(float(g.norm()) ** 2 for g in g_all))),
            'grad_norm_segmentation': gnorm('renderer.net.mlp_segmentation'),
            'grad_norm_encoder': gnorm('renderer.net.encoder'),
            'grad_norm_ray_attention': gnorm('renderer.net.ray_attention'),
            'nonfinite_grad_values': int(nonfinite),
        }
        model.zero_grad(set_to_none=True)
        print(label, out['samples'][label], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__':
    main()
