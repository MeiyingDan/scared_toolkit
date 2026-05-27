#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import tifffile
from tqdm import tqdm
from pathlib import Path
import scaredtk.io as sio
import pandas as pd

def load_generic_sample(p, scale_factor=256.0):
    if p.suffix=='.tiff':
        sample = tifffile.imread(str(p))
        if sample.shape[-1]==3:
            sample =sample[...,-1]
    else:
        sample = sio.load_subpix_png(p, scale_factor=scale_factor)
    
    return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_test_dir')
    parser.add_argument('root_prediction_dir')
    parser.add_argument('domain', choices=['disparity', 'depthmap'])
    parser.add_argument('--scale_factor', default=128.0, type=float)
    parser.add_argument(
        '--gt_domain',
        default=None,
        help='GT folder under keyframe_*/data/. Defaults to --domain (e.g. set depthmap_rectified).',
    )
    parser.add_argument(
        '--pred_domain',
        default=None,
        help='Prediction folder under keyframe_*/data/. Defaults to --domain.',
    )
    parser.add_argument(
        '--bad_threshold',
        type=float,
        default=None,
        help='If set, also report bad-threshold percentage over valid GT pixels.',
    )
    args = parser.parse_args()


    root_test_dir = Path(args.root_test_dir)
    root_prediction_dir = Path(args.root_prediction_dir)
    gt_domain = args.gt_domain if args.gt_domain is not None else args.domain
    pred_domain = args.pred_domain if args.pred_domain is not None else args.domain

    # search evaluation index files, that encode the names of frames with > 10%
    # coverage. If you not available, create them
    
    ref_keyframe_dirs = sorted([p for p in root_test_dir.rglob('**/keyframe_*') if p.is_dir()])
    pred_keyframe_dirs = sorted([p for p in root_prediction_dir.rglob('**/keyframe_*') if p.is_dir()])
    eval_lists = sorted([p for p in root_test_dir.rglob('**/valid.csv')])
    if len(eval_lists) != len(ref_keyframe_dirs):
        print('valid frame lists are missing, we need to generate them first')
        for kf in tqdm(ref_keyframe_dirs, desc='keyframes processed'):
            valid_list=[]
            depth_frames = sorted([p for p in (kf/'data'/gt_domain).iterdir()])
            if len(depth_frames)==0:
                print(f'cannot generate valid_list because {gt_domain} directory under keyframe_*/data/ does not exist', file=sys.stderr)
                return 1
            for frame_id, depthmap_path in tqdm(enumerate(depth_frames), total=len(depth_frames), desc='depthmap processed'):
                
                dmap = load_generic_sample(depthmap_path, args.scale_factor)
                    
                coverage = 1 - (np.count_nonzero(np.isnan(dmap))/dmap.size)
                if coverage>=.1:
                    valid_list.append(frame_id)
            np.savetxt(kf/"valid.csv", valid_list, fmt='%i', delimiter=",")
        eval_lists = sorted([p for p in root_test_dir.rglob('**/valid.csv')])
        
    assert len(eval_lists) == len(ref_keyframe_dirs)
    results_dataset=[]
    results_keyframe=[]
    results_mae=[]
    results_bad3=[]
    results_badx=[]
    for ref_kf, pred_kf in tqdm(zip(ref_keyframe_dirs, pred_keyframe_dirs), desc='keyframes processed', total= len(ref_keyframe_dirs)):
        assert ref_kf.name == pred_kf.name
        assert ref_kf.parent.name == pred_kf.parent.name
        try:
            valid_ids = list(np.loadtxt(ref_kf/"valid.csv", delimiter=',').astype(int))
        except TypeError:
            valid_ids = [int(np.loadtxt(ref_kf/"valid.csv", delimiter=','))]

        ref_paths = np.array(sorted([p for p in (ref_kf/'data'/gt_domain).iterdir()]))[valid_ids]
        pred_paths = np.array(sorted([p for p in (pred_kf/'data'/pred_domain).iterdir()]))[valid_ids]
          
        assert len(ref_paths) == len(pred_paths)
        mae_lst = []
        bad3_lst =[]
        badx_lst =[]
        
        for ref_p, pred_p in tqdm(zip(ref_paths, pred_paths), desc='samples', leave=False, total= len(ref_paths)):
            ref = load_generic_sample(ref_p, args.scale_factor)
            pred = load_generic_sample(pred_p, args.scale_factor)
            pred = np.nan_to_num(pred)
            
            error = np.abs(ref-pred)
            # we load zero disparity and depth ground truth values as nan
            # and ingnore them in the error computation
            mae_lst.append(np.nanmean(error))
            if args.domain == 'disparity':
                data_points = np.count_nonzero(~np.isnan(error))
                bad3_lst.append((np.sum(error>3)/data_points)*100)
            if args.bad_threshold is not None:
                valid = ~np.isnan(ref)
                data_points = np.count_nonzero(valid)
                if data_points > 0:
                    badx_lst.append((np.sum((error>args.bad_threshold) & valid)/data_points)*100)
                
        assert len(mae_lst) ==len(valid_ids)
        
        results_dataset.append(ref_kf.parent.name[-1])
        results_keyframe.append(ref_kf.name[-1])
        results_mae.append(np.mean(np.array(mae_lst)))
        if args.domain=='disparity':
            results_bad3.append(np.mean(np.array(bad3_lst)))
        if args.bad_threshold is not None and len(badx_lst) > 0:
            results_badx.append(np.mean(np.array(badx_lst)))
    
    results_dict = {'dataset':results_dataset, 'keyframe':results_keyframe, 'MAE':results_mae}
    if args.domain=='disparity':
        results_dict['bad3']=results_bad3
    if args.bad_threshold is not None and len(results_badx) == len(results_dataset):
        bad_col = f'bad{args.bad_threshold:g}'
        results_dict[bad_col]=results_badx
    
    results = pd.DataFrame(results_dict)
    
    results.to_csv(root_prediction_dir/f'results_{args.domain}.csv')
    txt = str(root_prediction_dir/f'results_{args.domain}.csv')
    print(f'results saves at: {txt}')
    return 0 
    
        
        
if __name__ == "__main__":
    sys.exit(main())