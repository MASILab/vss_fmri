# Vasculature-informed spatial smoothing for fMRI
Vasculature-informed spatial smoothing (VSS) filter for functional magnetic resonance imaging (fMRI) based on susceptibility-weighted imaging (SWI).

![vss_graphical_abstract](https://github.com/user-attachments/assets/e1ff8c87-1000-45c6-9863-4b31578a5cc1)

## Installation
Clone this repo and navigate to the downloaded directory. Use [conda](https://docs.conda.io/en/latest/) to create a Python environment with the specified requirements:

```bash
conda env create --name vss_fmri -f environment.yml
```

Install the vss_fmri package locally using pip:

```bash
pip install .
```

## Usage
To perform vasculature-informed spatial smoothing, you will need peak vasculature directions generated from SWI. We used a Frangi filter (see [an example MATLAB function here](https://www.mathworks.com/matlabcentral/fileexchange/24409-hessian-based-frangi-vesselness-filter)) to search for vessel-like structures and provide vasculature directions from the principal component of the Hessian filter. See our paper for details.

To use the vasculature-informed spatial smoothing filter, see the provided ```apply_vss_filter.py```.

```bash
data_dir="/path/to/data/dir"
wm_peaks="$data_dir/wm_peaks.nii.gz"
wm_mask="$data_dir/wm_mask.nii.gz"
input_fmri="$data_dir/fmri_reg_to_hcp.nii.gz"
output_fmri="$data_dir/fmri_filtered.nii.gz"
adj_matrix="$data_dir/adj_matrix_5x5x5_0.9.npz"

# Apply filter
python apply_vss_filter.py \
    --peaks $wm_peaks \
    --wm_mask $wm_mask \
    --adj_matrix $adj_matrix \
    --fmri_data $input_fmri \
    --output $output_fmri \
    --n 5 \
    --alpha 0.8 \
    --beta 50 \
    --n_jobs 15
```

We also provide code for diffusion-informed spatial smoothing (DSS), originally introduced by [Abramian et al](https://doi.org/10.1016/j.neuroimage.2021.118095). To use the diffusion-informed spatial smoothing filter, see ```apply_dss_filter.py```.

```bash
data_dir="/path/to/data/dir"
odf_sh="$data_dir/odf_sh.nii.gz"
wm_mask="$data_dir/wm_mask.nii.gz"
input_fmri="$data_dir/fmri_reg_to_hcp.nii.gz"
output_fmri="$data_dir/fmri_filtered.nii.gz"
adj_matrix="$data_dir/adj_matrix_5x5x5_0.9.npz"

# Apply filter
python apply_dss_filter.py \
    --odf_sh $odf_sh \
    --wm_mask $wm_mask \
    --adj_matrix $adj_matrix \
    --fmri_data $input_fmri \
    --output $output_fmri \
    --sh_format tournier \
    --n 5 \
    --alpha 0.8 \
    --beta 50 \
    --n_jobs 15
```

## Citation
The code is released under the MIT License.

If you use the filters in your research, please cite the following:

Vasculature-informed spatial smoothing filter:
> Adam M. Saunders, Michael E. Kim, Kurt G. Schilling, John C. Gore, Bennett A. Landman, and Yurui Gao. Vasculature-informed spatial smoothing of white matter functional magnetic resonance imaging. SPIE Medical Imaging: Image Processing, 2025, February, San Diego, California. [https://doi.org/10.1117/12.3047240](https://doi.org/10.1117/12.3047140).

Diffusion-informed spatial smoothing filter:
> David Abramian, Martin Larsson, Anders Eklund, Iman Aganj, Carl-Fredrik Westin, Hamid Behjat. Diffusion-informed spatial smoothing of fMRI data in white matter using spectral graph filters. NeuroImage, 2021. https://doi.org/10.1016/j.neuroimage.2021.118095.

