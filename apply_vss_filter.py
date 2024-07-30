# Script to create DSS filter and save
import os 
import argparse
import numpy as np
import nibabel as nib
import vss_fmri
import scipy.sparse as sp

# Set up argparse
parser = argparse.ArgumentParser(description='Create DSS filter and apply to fMRI data')
parser.add_argument('--peaks', type=str, help='Path to peak SWI unit vector directions in a NIFTI file, should be r x c x d x 3')
parser.add_argument('--wm_mask', type=str, help='Path to WM mask NIFTI file')
parser.add_argument('--fmri_data', type=str, help='Path to fMRI data NIFTI file')
parser.add_argument('--output', type=str, help='Path to filtered fMRI data save location')
parser.add_argument('--adj_matrix', type=str, help='Path to adjacency matrix for VSS filter')
parser.add_argument('--n', type=int, help='Size of filter (3 or 5), default=5', default=5)
parser.add_argument('--alpha', type=float, help='Alpha parameter for filter, default=0.9', default=0.9)
parser.add_argument('--beta', type=float, help='Beta parameter for filter, default=50', default=50)
parser.add_argument('--n_jobs', type=int, help='CPU jobs for parallel processing, default=1', default=1)

# Parse arguments
args = parser.parse_args()
peaks = args.peaks
wm_mask = args.wm_mask
fmri_data = args.fmri_data
fmri_filtered_output = args.output
adj_matrix_path = args.adj_matrix
n = args.n
alpha = args.alpha
beta = args.beta
n_jobs = args.n_jobs

# Load data
print('Creating VSS filter...')
peaks = nib.load(peaks)
peaks = nib.funcs.as_closest_canonical(peaks)
wm_mask = nib.load(wm_mask)
wm_mask = nib.funcs.as_closest_canonical(wm_mask)
vss_filter = vss_fmri.graph.VSSFilter(peaks, wm_mask, adj_matrix_path, n=n, alpha=alpha, beta=beta)
print('Done!')

# Apply to fMRI data
fmri_data = nib.load(fmri_data)
fmri_data = nib.funcs.as_closest_canonical(fmri_data)
fmri_filt = vss_filter.apply_filter(fmri_data, deg=15, lambda_min=0, lambda_max=2, tau=7, n_jobs=n_jobs)
print('Saving filtered data...')
fmri_filt.to_filename(fmri_filtered_output)
print('Filtered data saved to', fmri_filtered_output)
