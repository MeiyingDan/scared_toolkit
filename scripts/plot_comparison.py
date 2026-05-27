import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import tifffile
from tqdm import tqdm
import scaredtk.io as sio

def load_generic_sample(p, scale_factor=256.0):
    """Load depthmap or disparity from file"""
    if p.suffix=='.tiff':
        sample = tifffile.imread(str(p))
        if sample.shape[-1]==3:
            sample = sample[...,-1]
    else:
        sample = sio.load_subpix_png(p, scale_factor=scale_factor)
    return sample

def compute_per_frame_errors(
    ground_truth_dir,
    prediction_dirs,
    domain='depthmap',
    gt_domain=None,
    pred_domain=None,
    scale_factor=128.0
):
    """
    Compute per-frame errors for multiple methods
    
    Args:
        ground_truth_dir: Path to SCARED_DATASET_processed
        prediction_dirs: Dict of {method_name: prediction_dir_path}
        domain: Backward-compatible default domain for both GT and prediction
        gt_domain: Optional GT subfolder under keyframe_*/data/
        pred_domain: Optional prediction subfolder under keyframe_*/data/
        scale_factor: Scale factor for loading
    
    Returns:
        Dict of {(dataset, keyframe): {method: per_frame_errors}}
    """
    gt_domain = gt_domain if gt_domain is not None else domain
    pred_domain = pred_domain if pred_domain is not None else domain
    results = defaultdict(lambda: defaultdict(list))
    
    # Find all keyframes in ground truth
    gt_path = Path(ground_truth_dir)
    datasets = sorted([d for d in gt_path.iterdir() if d.is_dir() and d.name.startswith('dataset_')])
    
    for dataset_dir in datasets:
        dataset_name = dataset_dir.name
        keyframes = sorted([k for k in dataset_dir.iterdir() 
                          if k.is_dir() and k.name.startswith('keyframe_')])
        
        for keyframe_dir in keyframes:
            keyframe_name = keyframe_dir.name
            key = (dataset_name, keyframe_name)
            
            # Load valid frame indices
            valid_csv = keyframe_dir / 'valid.csv'
            if valid_csv.exists():
                try:
                    valid_arr = np.loadtxt(valid_csv, delimiter=',')
                    if valid_arr.ndim == 0:  # single value
                        valid_ids = [int(valid_arr)]
                    else:
                        valid_ids = valid_arr.astype(int).tolist()
                except:
                    continue
            else:
                continue
            
            # Get ground truth files
            gt_domain_dir = keyframe_dir / 'data' / gt_domain
            gt_files = sorted([f for f in gt_domain_dir.iterdir() if f.suffix in ['.png', '.tiff']])
            gt_files = np.array(gt_files)[valid_ids]
            
            # For each method, compute per-frame errors
            for method_name, pred_dir in prediction_dirs.items():
                # Find prediction files for this dataset/keyframe
                pred_keyframe_dir = pred_dir / dataset_name / keyframe_name / 'data' / pred_domain
                
                if not pred_keyframe_dir.exists():
                    print(f'Warning: {pred_keyframe_dir} not found, skipping {method_name}')
                    continue
                
                pred_files = sorted([f for f in pred_keyframe_dir.iterdir() if f.suffix in ['.png', '.tiff']])
                pred_files = np.array(pred_files)[valid_ids]
                
                # Compute per-frame MAE
                per_frame_mae = []
                for gt_file, pred_file in tqdm(zip(gt_files, pred_files), 
                                               total=len(gt_files),
                                               desc=f'{method_name}/{dataset_name}/{keyframe_name}',
                                               leave=False):
                    gt = load_generic_sample(gt_file, scale_factor)
                    pred = load_generic_sample(pred_file, scale_factor)
                    pred = np.nan_to_num(pred)
                    
                    error = np.abs(gt - pred)
                    mae = np.nanmean(error)
                    per_frame_mae.append(mae)
                
                results[key][method_name] = per_frame_mae
    
    return results

def plot_results(results, output_dir, domain='depthmap'):
    """Generate plots for all dataset/keyframe combinations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams.update({'font.size': 22})
    
    for (dataset_name, keyframe_name), methods_data in sorted(results.items()):
        # Create figure
        fig, ax = plt.subplots(figsize=(40, 14))
        
        # Plot each method
        for method_name, errors in sorted(methods_data.items()):
            ax.plot(errors, linewidth=3, label=method_name.replace('_', ' ').title())
        
        # Customize plot
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Mean Error (mm)')
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        
        # Add legend
        ax.legend(loc='best')
        
        # Save figure
        output_subdir = output_dir / dataset_name / keyframe_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        output_path_pdf = output_subdir / 'plot.pdf'
        output_path_png = output_subdir / 'plot.png'
        
        plt.savefig(output_path_pdf, bbox_inches='tight', dpi=100)
        plt.savefig(output_path_png, bbox_inches='tight', dpi=100)
        print(f'Saved plot to {output_path_pdf}')
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Generate per-frame comparison plots for stereo evaluation'
    )
    parser.add_argument(
        'ground_truth_dir',
        help='Path to SCARED_DATASET_processed'
    )
    parser.add_argument(
        'method_dirs',
        nargs='+',
        help='Prediction directories (each should contain dataset_X/keyframe_Y/data/{domain}/ structure)'
    )
    parser.add_argument(
        '--domain',
        default='depthmap',
        choices=['depthmap', 'disparity'],
        help='Domain to evaluate'
    )
    parser.add_argument(
        '--output_dir',
        default='./plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--gt_domain',
        default=None,
        help='GT subfolder under keyframe_*/data/. Defaults to --domain.'
    )
    parser.add_argument(
        '--pred_domain',
        default=None,
        help='Prediction subfolder under keyframe_*/data/. Defaults to --domain.'
    )
    parser.add_argument(
        '--method_names',
        nargs='+',
        help='Names of methods (if not provided, parent directory names will be used)'
    )
    parser.add_argument(
        '--scale_factor',
        type=float,
        default=128.0,
        help='Scale factor for loading depthmap/disparity'
    )
    args = parser.parse_args()
    
    # Parse method directories
    prediction_dirs = {}
    for idx, method_dir in enumerate(args.method_dirs):
        if args.method_names and idx < len(args.method_names):
            method_name = args.method_names[idx]
        else:
            method_name = Path(method_dir).parent.name
        
        prediction_dirs[method_name] = Path(method_dir)
    
    print('Computing per-frame errors...')
    results = compute_per_frame_errors(
        args.ground_truth_dir,
        prediction_dirs,
        domain=args.domain,
        gt_domain=args.gt_domain,
        pred_domain=args.pred_domain,
        scale_factor=args.scale_factor
    )
    
    print('Generating plots...')
    plot_results(results, args.output_dir, domain=args.domain)
    
    print(f'All plots saved to {args.output_dir}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
