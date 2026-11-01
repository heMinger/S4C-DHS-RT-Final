#!/usr/bin/env python
"""CPU check: map old generator-view DA3 depth/conf into the training fisheye RGB grid.

Frame: image_02 / 0000006447. Compares, at the same physical LiDAR points:
  - original training-view resize (current bug)
  - generator-view reference
  - adapted training-view depth (pixel mapping + z->ray-distance conversion)
and runs the DHS 32+32/fallback sampling on the same training pixels.
No model, no GPU, no DA3 regeneration.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2, yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def stats(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not x.size: return {'n':0}
    return {'n':int(x.size),'mean':float(x.mean()),'p50':float(np.median(x)),
            'p05_p95':np.percentile(x,[5,95]).tolist()}

def dm(pred,truth):
    pred=np.asarray(pred,float); truth=np.asarray(truth,float)
    v=np.isfinite(pred)&np.isfinite(truth)&(pred>0)&(truth>0)
    pred,truth=pred[v],truth[v]
    if pred.size<20: return {'n':int(pred.size),'status':'insufficient'}
    rel=np.abs(pred-truth)/truth; ratio=np.maximum(pred/truth,truth/pred)
    return {'n':int(pred.size),'abs_rel':float(rel.mean()),'median_abs_rel':float(np.median(rel)),
            'rmse':float(np.sqrt(np.mean((pred-truth)**2))),
            'median_pred_over_gt':float(np.median(pred/truth)),
            'delta_1_25':float(np.mean(ratio<1.25))}

def fallback_reason(d,c):
    d=np.asarray(d,float); c=np.asarray(c,float)
    norm=np.clip((c-1.)/5.5535,.1,1.)
    invalid=(d<=0)|(d<3.01)
    low=norm<.2
    fb=invalid|low
    return {'n':int(d.size),'invalid_depth':int(invalid.sum()),'low_conf':int(low.sum()),
            'both':int((invalid&low).sum()),'fallback':int(fb.sum()),'active':int((~fb).sum()),
            'fallback_pct':float(fb.mean()*100),'active_pct':float((~fb).mean()*100),
            'depth_positive_pct':float((d>0).mean()*100),
            'conf_p05_p50_p95':np.percentile(c,[5,50,95]).tolist() if c.size else None,
            'norm_conf_p05_p50_p95':np.percentile(norm,[5,50,95]).tolist() if norm.size else None}

def nearest_mask(pid,radius,valid):
    out=np.zeros(radius.shape,bool); ids=np.flatnonzero(valid)
    if ids.size:
        order=np.lexsort((ids,radius[ids],pid[ids])); s=ids[order]
        first=np.r_[True,np.diff(pid[s])!=0]; out[s[first]]=True
    return out

def raw_project(xyz,calib,use_distortion=True):
    radius=np.linalg.norm(xyz,axis=1); u=xyz/np.maximum(radius[:,None],1e-9)
    xi=calib['mirror_parameters']['xi']
    q=u[:,:2]/(u[:,2:3]+xi)
    if use_distortion:
        d=calib['distortion_parameters']; r=q[:,0]**2+q[:,1]**2
        f=1+d['k1']*r+d['k2']*r*r
        p1,p2=d.get('p1',0),d.get('p2',0)
        x=q[:,0]*f+2*p1*q[:,0]*q[:,1]+p2*(r+2*q[:,0]**2)
        y=q[:,1]*f+p1*(r+2*q[:,1]**2)+2*p2*q[:,0]*q[:,1]
        q=np.stack([x,y],axis=1)
    p=calib['projection_parameters']
    return np.stack([p['gamma1']*q[:,0]+p['u0'],p['gamma2']*q[:,1]+p['v0']],axis=1),radius

def sample_dhs(d,c,rng,n_surface=32,n_global=32,near=3.,far=80.,absrel=.12,min_th=.5,eps=.1,thr=.2):
    norm=np.clip((c-1.)/5.5535,eps,1.)
    invalid=(d<=0)|(d<near+.01)
    low=norm<thr
    fb=invalid|low
    d=np.clip(d,near,far)
    sigma=np.maximum(min_th,d/ np.maximum(norm,1e-9)*absrel)
    lo=np.maximum(d-2*sigma,near); hi=np.minimum(d+2*sigma,far)
    ns=np.arange(n_surface)[None,:]
    ss=(ns+rng.random((len(d),n_surface)))/n_surface
    surf=lo[:,None]+ss*((hi-lo)[:,None])
    ng=np.arange(n_global)[None,:]
    gs=(ng+rng.random((len(d),n_global)))/n_global
    glob=1/((1-gs)/near+gs/far)
    allp=np.sort(np.concatenate([surf,glob],axis=1),axis=1)
    if fb.any():
        n=len(d[fb]); ns64=(np.arange(64)[None,:]+rng.random((n,64)))/64
        allp[fb]=np.sort(1/((1-ns64)/near+ns64/far),axis=1)
    return allp,fb

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo', default='/home/lmh/s4c-fix-data-20261101')
    ap.add_argument('--root', default='/mnt/disk1/lmh/KITTI360')
    ap.add_argument('--seq', default='2013_05_28_drive_0004_sync')
    ap.add_argument('--frame', default='0000006447')
    ap.add_argument('--cam', default='image_02')
    ap.add_argument('--max-lidar', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--out', required=True)
    a=ap.parse_args()
    sys.path.insert(0,a.repo)
    from datasets.kitti_360.kitti_360_dataset import Kitti360Dataset, FisheyeToPinholeSampler
    root=Path(a.root); rel=Path(a.seq)/a.cam; fid=f'{int(a.frame):010d}.png'
    raw_path=root/'data_2d_raw'/rel/'data_rgb'/fid
    train_path=root/'KITTI360_preprocessed/KITTI360_preprocessed/data_2d_raw'/rel/'data_192x640_0x-15'/fid
    depth_path=root/'data_2d_depth_da3_v3'/rel/'data_rect'/fid
    conf_path=root/'data_2d_depth_da3_v3'/rel/'data_rect_conf'/fid
    lidar_path=root/'data_3d_raw'/a.seq/'velodyne_points/data'/fid.replace('.png','.bin')
    raw=cv2.imread(str(raw_path)); train_rgb=cv2.imread(str(train_path))
    depth_raw=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
    conf_raw=cv2.imread(str(conf_path),cv2.IMREAD_UNCHANGED)
    if raw is None or train_rgb is None or depth_raw is None or conf_raw is None:
        raise SystemExit('missing image/depth/conf')
    depth=depth_raw.astype(np.float64)/256.
    conf=np.clip(cv2.resize(conf_raw.astype(np.float32)/10000.,(depth.shape[1],depth.shape[0])),0,6.5535) if conf_raw.shape!=depth.shape else np.clip(conf_raw.astype(np.float32)/10000.,0,6.5535)
    h,w=depth.shape
    cal=Kitti360Dataset._load_calibs(str(root),(0,-15))
    cam_idx=int(a.cam[-3+1:]) if False else int(a.cam.split('_')[-1])
    sampler=FisheyeToPinholeSampler(cal['K_fisheye'],(192,640),cal['fisheye'][f'calib_{cam_idx:02d}'],cal['fisheye'][f'R_{cam_idx:02d}'])
    lm=np.asarray(sampler.sample_pts.detach().cpu().numpy())
    if lm.ndim==4: lm=lm[0]
    raw_h,raw_w=raw.shape[:2]
    lx=((lm[...,0]+1)*.5*(raw_w-1)).astype(np.float64)
    ly=((lm[...,1]+1)*.5*(raw_h-1)).astype(np.float64)
    lmap=np.stack([lx,ly],axis=-1)
    # generator input view: 376x1408 pinhole, Ry(+15), no radial distortion, then depth-grid resize
    perspective={}
    for line in (root/'calibration/perspective.txt').read_text().splitlines():
        k,_,v=line.partition(':')
        if k=='P_rect_00': perspective[k]=np.array([float(x) for x in v.split()]).reshape(3,4)
    Kgen=perspective['P_rect_00'][:,:3]
    original=yaml.safe_load((root/f'calibration/{a.cam}.yaml').read_text().replace('%YAML:1.0',''))
    Rg=np.linalg.inv(Rotation.from_euler('y',-15,degrees=True).as_matrix())
    Rl=np.asarray(cal['fisheye'][f'R_{cam_idx:02d}'])
    xg,yg=np.meshgrid(np.arange(w),np.arange(h))
    # depth grid normalized over the generator 376x1408 input view
    xn=(xg+.5)/w*2-1; yn=(yg+.5)/h*2-1
    ug=(xn+1)*.5*(1408-1); vg=(yn+1)*.5*(376-1)
    q=np.linalg.inv(Kgen)@np.stack([ug.ravel(),vg.ravel(),np.ones(ug.size)])
    rg=Rg@q; rg/=np.linalg.norm(rg,axis=0,keepdims=True)
    xi=original['mirror_parameters']['xi']; qq=rg[:2]/(rg[2]+xi)
    pp=original['projection_parameters']
    gx=(pp['gamma1']*qq[0]+pp['u0']).reshape(h,w).astype(np.float64)
    gy=(pp['gamma2']*qq[1]+pp['v0']).reshape(h,w).astype(np.float64)
    gmap=np.stack([gx,gy],axis=-1)
    cos_g=rg[2].reshape(h,w)
    # training pixel -> nearest generator-depth pixel
    tree=cKDTree(gmap.reshape(-1,2))
    dist,pid=tree.query(lmap.reshape(-1,2))
    dist=dist.reshape(192,640); pid=pid.reshape(192,640)
    valid_map=dist<=3.0
    py,px=pid//w,pid%w
    z_adapt=np.where(valid_map,depth[py,px],0.)
    t_adapt=np.where(valid_map,depth[py,px]/np.maximum(cos_g[py,px],1e-6),0.)
    c_adapt=np.where(valid_map,conf[py,px],0.)
    orig_depth=cv2.resize(depth.astype(np.float32),(640,192))
    orig_conf=cv2.resize(conf.astype(np.float32),(640,192))
    result={'scope':'one frame image_02/6447; CPU only; no model/DA3',
            'loader_sha256':__import__('hashlib').sha256((Path(a.repo)/'datasets/kitti_360/kitti_360_dataset.py').read_bytes()).hexdigest(),
            'depth_shape':[h,w],'mapping':{'valid_pct':float(valid_map.mean()*100),
            'source_error_px':stats(dist),'fallback_pct':float((~valid_map).mean()*100)},
            'depth_quantity':{}}
    # LiDAR -> raw; training/generator pixel lookups
    data=np.fromfile(lidar_path,dtype=np.float32); pts=data.reshape(-1,4).astype(np.float64); pts[:,3]=1
    T=cal['T_cam_to_pose'][f'{cam_idx:02d}'].copy(); T[:3,:3]=T[:3,:3]@np.linalg.inv(Rl)
    Tvelo_raw=np.linalg.inv(T)@cal['T_velo_to_pose']
    xyz=(Tvelo_raw@pts.T)[:3]; radius=np.linalg.norm(xyz,axis=0)
    raw_uv,radius=raw_project(xyz.T,original,True)
    finite=np.isfinite(xyz).all(axis=0)&(radius>=3)&(radius<=80)&(raw_uv[:,0]>=0)&(raw_uv[:,0]<=raw_w-1)&(raw_uv[:,1]>=0)&(raw_uv[:,1]<=raw_h-1)&(xyz[2]>0)
    # training-view lookup
    ltree=cKDTree(lmap.reshape(-1,2)); lerr,lpid=ltree.query(raw_uv)
    lpy,lpx=lpid//640,lpid%640
    z_lidar=(np.linalg.inv(Rl)@xyz)[2]
    z_lidar_gen=(np.linalg.inv(Rg)@xyz)[2]
    lvalid=finite&(lerr<=3)
    lvis=nearest_mask(lpid,radius,lvalid); lvis&=np.isfinite(orig_depth[lpy,lpx])&(orig_depth[lpy,lpx]>0)
    # generator-view reference
    gtree=cKDTree(gmap.reshape(-1,2)); gerr,gpid=gtree.query(raw_uv)
    gpy,gpx=gpid//w,gpid%w
    gz=depth[gpy,gpx]
    gvalid=finite&(gerr<=3)&np.isfinite(gz)&(gz>0)
    gvis=nearest_mask(gpid,radius,gvalid)
    result['coverage']={'training_valid':int(lvalid.sum()),'training_visible':int(lvis.sum()),
                        'generator_valid':int(gvalid.sum()),'generator_visible':int(gvis.sum()),
                        'training_lookup_error_px':stats(lerr[lvalid]),
                        'generator_lookup_error_px':stats(gerr[gvalid])}
    mask=lvis&gvis
    # restrict to a representative cap
    idx=np.flatnonzero(mask)
    if idx.size>a.max_lidar:
        idx=np.random.default_rng(a.seed).choice(idx,a.max_lidar,replace=False)
    result['depth_quantity']={
        'training_original_Z':dm(orig_depth[lpy[idx],lpx[idx]],z_lidar[idx]),
        'training_original_t':dm(orig_depth[lpy[idx],lpx[idx]],radius[idx]),
        'generator_reference_Z':dm(depth[gpy[idx],gpx[idx]],z_lidar_gen[idx]),
        'generator_reference_t':dm(depth[gpy[idx],gpx[idx]],radius[idx]),
        'adapted_Z_valid':dm(z_adapt[lpy[idx],lpx[idx]],z_lidar_gen[idx]),
        'adapted_t_valid':dm(t_adapt[lpy[idx],lpx[idx]],radius[idx]),
        'adapted_coverage_on_paired':float(valid_map[lpy[idx],lpx[idx]].mean()*100),
    }
    # DHS sampling on paired points with adapted-valid support
    sel=idx[valid_map[lpy[idx],lpx[idx]]]
    if sel.size>5000: sel=np.random.default_rng(a.seed+1).choice(sel,5000,replace=False)
    rng=np.random.default_rng(a.seed+2)
    def dhs_eval(d,c):
        allp,fb=sample_dhs(d,c,rng)
        e=np.min(np.abs(allp-radius[sel,None]),axis=1)
        return {'n':int(sel.size),'active_pct':float((~fb).mean()*100),'fallback_pct':float(fb.mean()*100),
                'nearest_m':stats(e),'hit_0.2_pct':float((e<=.2).mean()*100),'hit_0.5_pct':float((e<=.5).mean()*100)}
    # ---- denominator diagnostics: full 192x640 vs selected LiDAR/paired subset ----
    funnel={'full_pixels':int(192*640),
            'lidar_finite':int(finite.sum()),
            'training_valid':int(lvalid.sum()),
            'training_visible':int(lvis.sum()),
            'generator_valid':int(gvalid.sum()),
            'generator_visible':int(gvis.sum()),
            'paired_visible':int(mask.sum()),
            'paired_adapted_valid':int((mask & valid_map[lpy,lpx]).sum()),
            'selected_after_cap':int(sel.size),
            'cap':int(a.max_lidar)}
    result['diagnostics']={
        'selection_funnel':funnel,
        'formulas':{
            'valid_map':'nearest distance in raw 1400x1400 fisheye pixels from training pixel raw coord to generator-depth raw coord <= 3.0',
            'z_adapt':'depth[py,px] (raw DA3 value at nearest generator-depth pixel; passed to sampler as t)',
            't_adapt':'depth[py,px] / max(cos_g[py,px],1e-6), where cos_g is generator virtual pinhole unit-ray z',
            'c_adapt':'conf[py,px]; invalid mapping -> depth=conf=0',
            'dhs_fallback':'depth<=0 OR depth<3.01 OR norm_conf<0.2; norm_conf=clip((conf-1)/5.5535,0.1,1.0)',
            'name_warning':'adapted_z means raw-depth-as-sampler-t; adapted_t means raw/cos_g-as-sampler-t',
        },
        'full_frame_original':fallback_reason(orig_depth.reshape(-1),orig_conf.reshape(-1)),
        'full_frame_adapted_raw_as_t':fallback_reason(z_adapt.reshape(-1),c_adapt.reshape(-1)),
        'selected_subset_original':fallback_reason(orig_depth[lpy[sel],lpx[sel]],orig_conf[lpy[sel],lpx[sel]]),
        'selected_subset_adapted_raw_as_t':fallback_reason(z_adapt[lpy[sel],lpx[sel]],c_adapt[lpy[sel],lpx[sel]]),
        'reference_dense_fallback_pct':75.27180989583333,
        'reference_source':'historical_da3_lidar_server_result_20261031.json (image_02/6447, all 192x640 pixels)',
    }
    result['dhs']={'original':dhs_eval(orig_depth[lpy[sel],lpx[sel]],orig_conf[lpy[sel],lpx[sel]]),
                   'adapted_t':dhs_eval(t_adapt[lpy[sel],lpx[sel]],c_adapt[lpy[sel],lpx[sel]]),
                   'adapted_z':dhs_eval(z_adapt[lpy[sel],lpx[sel]],c_adapt[lpy[sel],lpx[sel]])}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
