# Script to create DSS filter and save
import argparse
import nibabel as nib
import vss_fmri

# Set up argparse
parser = argparse.ArgumentParser(description='Create DSS filter and apply to fMRI data')
parser.add_argument('--odf_sh', type=str, help='Path to ODF NIFTI file in spherical harmonics')
parser.add_argument('--wm_mask', type=str, help='Path to WM mask NIFTI file')
parser.add_argument('--fmri_data', type=str, help='Path to fMRI data NIFTI file')
parser.add_argument('--output', type=str, help='Path to filtered fMRI data save location')
parser.add_argument('--adj_matrix', type=str, help='Path to adjacency matrix for DSS filter')
parser.add_argument('--sh_format', type=str, help='Spherical harmonic basis format (tournier or descoteaux, see https://workshop.dipy.org/documentation/1.5.0/theory/sh_basis/), default=tournier', default='tournier')
parser.add_argument('--n', type=int, help='Size of neighborhood for filter (3 or 5), default=5', default=5)
parser.add_argument('--alpha', type=float, help='Alpha parameter for filter, default=0.9', default=0.9)
parser.add_argument('--beta', type=float, help='Beta parameter for filter, default=50', default=50)
parser.add_argument('--n_jobs', type=int, help='CPU jobs for parallel processing, default=1', default=1)

# Parse arguments
args = parser.parse_args()
odf_sh = args.odf_sh
wm_mask = args.wm_mask
fmri_data = args.fmri_data
fmri_filtered_output = args.output
adj_matrix_path = args.adj_matrix
sh_format = args.sh_format
n = args.n
alpha = args.alpha
beta = args.beta
n_jobs = args.n_jobs

# Load data
print('Creating DSS filter...')
odf_sh = nib.load(odf_sh)
wm_mask = nib.load(wm_mask)
dss_filter = vss_fmri.graph.DSSFilter(odf_sh, wm_mask, adj_matrix_path, n=n, alpha=alpha, beta=beta, sh_method=sh_format)
print('Done!')

# Apply to fMRI data
fmri_data = nib.load(fmri_data)
fmri_filt = dss_filter.apply_filter(fmri_data, deg=15, lambda_min=0, lambda_max=2, tau=7, n_jobs=n_jobs)
print('Saving filtered data...')
fmri_filt.to_filename(fmri_filtered_output)
print('Filtered data saved to', fmri_filtered_output)
