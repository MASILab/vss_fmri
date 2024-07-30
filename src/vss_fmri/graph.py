import os
import numpy as np
import nibabel as nib
import scipy.sparse as sp
from tqdm import tqdm
from .utils import  tunable_sigmoid, get_neighbors, create_sampling_template, sample_odfs, get_chebyshev_coeff, apply_filter
from joblib import Parallel, delayed
from functools import cached_property

class DSSFilter():
    def __init__(self, odfs_sh, wm_mask, adj_matrix_file, n=5, alpha=0.9, beta=50, template=None,
        sh_method='tournier', sh_order=8, threshold=None):
        """Class to create and apply diffusion-informed spatial smoothing filter for fMRI.
        Code based on https://github.com/DavidAbramian/DSS/tree/master

        Parameters
        ----------
        odfs_sh : nibabel.Nifti1Image
            Spherical harmonics, stored in r x c x d x 45 for order 8 (Tournier/mrtrix convention)
        wm_mask : nibabel.Nifti1Image
            White matter mask to define voxels.
        adj_matrix_file : str
            Path to adjacency matrix .npz. Can be saved or path to save to.
        n : int, optional
            Number of connected neighbors, by default 5
        alpha : float, optional
            alpha parameter for sigmoid function, by default 0.9
        beta : int, optional
            beta parameter for sigmoid function, by default 50
        template : (M,3) array, optional
            Template to use for sampling, by default None will create a sampling template
            from an icosahedron. Should be M directions covering a solid angle 4*pi/M.
        sh_method : str, optional
            Spherical harmonics method, by default 'tournier'. Can also be 'descoteaux'.
        sh_order : int, optional
            Spherical harmonics order, by default 8
        threhsold : float, optional
            Threshold to apply to adjacency matrix, by default None

        Attributes
        ----------
        self.adj_matrix : scipy.sparse.csc.csc_matrix
            CSC sparse coherence-weighted adjacency matrix 
        self.unweighted_adj_matrix : scipy.sparse.csc.csc_matrix
            CSC sparse unweighted adjacency matrix
        self.subscript_to_linear_index : np.ndarray
            Mapping from voxel subscript to linear index. Linear index at x,y,z is subscript_to_linear_index[x,y,z]
        self.linear_to_subscript_index : np.ndarray
            Mapping from linear index to voxel subscript. Subscript at linear index i is linear_to_subscript_index[i]

        Methods
        -------
        apply_filter(fmri_data, deg=15, lambda_min=0, lambda_max=2, tau=7, n_jobs=18)
            Apply filter to fMRI data.
        """
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.adj_matrix_file = adj_matrix_file
        if template is None:
            self.template = create_sampling_template(n)
        else:
            self.template = template
        self.sh_method = sh_method
        self.sh_order = sh_order
        self.threshold = threshold

        # Load odfs and mask
        odfs_sh = nib.funcs.as_closest_canonical(odfs_sh)
        wm_mask = nib.funcs.as_closest_canonical(wm_mask)
        self.odfs_sh = odfs_sh.get_fdata()
        self.wm_mask = wm_mask.get_fdata()

        # Get subscript to linear voxel and vice versa
        wm_mask_flat = np.ravel(self.wm_mask)
        I_mask = np.flatnonzero(wm_mask_flat)
        nonzero_voxels = np.argwhere(self.wm_mask > 0)
        self.n_voxels = nonzero_voxels.shape[0]
        I_vox = np.zeros_like(self.wm_mask, dtype=int)
        I_vox_flat = I_vox.flatten()
        I_vox_flat[I_mask] = np.arange(self.n_voxels)
        I_vox = I_vox_flat.reshape(self.wm_mask.shape)
        self.subscript_to_linear_index = I_vox

        I_ind = np.zeros((self.n_voxels, 3), dtype=int)
        I_ind[:,0], I_ind[:,1], I_ind[:,2] = np.unravel_index(I_mask, wm_mask.shape)
        self.linear_to_subscript_index = I_ind

    @cached_property
    def unweighted_adj_matrix(self):
        # Create unweighted adjacency matrix
        neighbors = get_neighbors(self.n)
        mask_voxels = np.argwhere(self.wm_mask)
        mask_indices = self.subscript_to_linear_index[mask_voxels[:,0], mask_voxels[:,1], mask_voxels[:,2]]
        column_indices = np.repeat(mask_indices, neighbors.shape[0])
        column_voxels = self.linear_to_subscript_index[column_indices]
        row_voxels = np.tile(neighbors, (self.n_voxels, 1))#np.tile(mask_voxels, (1, neighbors.shape[0]))
        row_voxels += np.repeat(mask_voxels, neighbors.shape[0], axis=0)
        row_indices = self.subscript_to_linear_index[row_voxels[:,0], row_voxels[:,1], row_voxels[:,2]]

        # Get valid indices
        valid_row_indices = np.all((row_voxels >= 0) & (row_voxels < self.wm_mask.shape), axis=1)
        valid_column_indices = np.all((column_voxels >= 0) & (column_voxels < self.wm_mask.shape), axis=1)
        valid_voxels = valid_row_indices & valid_column_indices

        # Check if they are in WM mask
        valid_row_indices = valid_row_indices & self.wm_mask[row_voxels[:,0], row_voxels[:,1], row_voxels[:,2]].astype(bool)
        valid_column_indices = valid_column_indices & self.wm_mask[column_voxels[:,0], column_voxels[:,1], column_voxels[:,2]].astype(bool)
        valid_indices = valid_voxels & valid_row_indices & valid_column_indices

        # Create adjacency matrix
        data = np.ones_like(valid_indices, dtype=np.int32)
        adj_matrix = sp.csc_array((data[valid_indices], (row_indices[valid_indices], column_indices[valid_indices])), shape=(self.n_voxels, self.n_voxels))
        return adj_matrix

    @cached_property
    def adj_matrix(self):
        # Check if adj_matrix_file exists
        if os.path.exists(self.adj_matrix_file):
            print(f'Loading {self.adj_matrix_file}...')
            adj_matrix = sp.load_npz(self.adj_matrix_file)

            # Remove NaN
            adj_matrix.data = np.nan_to_num(adj_matrix.data)
            adj_matrix.data[adj_matrix.data == np.inf] = 0
            print('Done!')
            return adj_matrix

        # Create adjacency matrix
        print(f'{self.adj_matrix_file} not found. Creating adjacency matrix...')

        # Create unweighted adjacency matrix
        neighbors = get_neighbors(self.n)
        mask_voxels = np.argwhere(self.wm_mask)
        mask_indices = self.subscript_to_linear_index[mask_voxels[:,0], mask_voxels[:,1], mask_voxels[:,2]]
        column_indices = np.repeat(mask_indices, neighbors.shape[0])
        column_voxels = self.linear_to_subscript_index[column_indices]
        row_voxels = np.tile(neighbors, (self.n_voxels, 1))
        row_voxels += np.repeat(mask_voxels, neighbors.shape[0], axis=0)
        valid_row_indices = np.all((row_voxels >= 0) & (row_voxels < self.wm_mask.shape), axis=1)
        valid_column_indices = np.all((column_voxels >= 0) & (column_voxels < self.wm_mask.shape), axis=1)
        valid_voxels = valid_row_indices & valid_column_indices
        row_voxels[~valid_voxels] = 0
        column_voxels[~valid_voxels] = 0

        column_indices = self.subscript_to_linear_index[column_voxels[:,0], column_voxels[:,1], column_voxels[:,2]]
        row_indices = self.subscript_to_linear_index[row_voxels[:,0], row_voxels[:,1], row_voxels[:,2]]

        # Check if they are in WM mask
        valid_row_indices = valid_row_indices & self.wm_mask[row_voxels[:,0], row_voxels[:,1], row_voxels[:,2]].astype(bool)
        valid_column_indices = valid_column_indices & self.wm_mask[column_voxels[:,0], column_voxels[:,1], column_voxels[:,2]].astype(bool)
        valid_indices = valid_voxels & valid_row_indices & valid_column_indices
    
        # Compute weighted adjacency matrix
        norm_dirs = neighbors / np.linalg.norm(neighbors, axis=1)[:, None]

        # Reshape ODFs to n_voxels x 45
        odfs_sh = self.odfs_sh[self.wm_mask.astype(bool), :].T
        data = sample_odfs(norm_dirs, self.template, odfs_sh, sh_method=self.sh_method, sh_order=self.sh_order).T

        # Normalize to [0, 0.5]
        max_dir = np.max(data, axis=1)
        max_dir[max_dir == 0] = 1
        data_reshape = 0.5*data / max_dir[:, None]
        data = data_reshape.flatten()

        # Create adjacency matrix
        adj_matrix = sp.csc_array((data[valid_indices], (row_indices[valid_indices], column_indices[valid_indices])), shape=(self.n_voxels, self.n_voxels))

        # Add to transpose
        adj_matrix = adj_matrix + adj_matrix.T

        # Apply tunable sigmoid
        weights = tunable_sigmoid(adj_matrix.data, alpha=self.alpha, beta=self.beta)
        adj_matrix.data = adj_matrix.data * weights

        # Threshold values below 1e-10
        if self.threshold is not None:
            adj_matrix.data[adj_matrix.data < self.threshold] = 0
            adj_matrix.eliminate_zeros()

        sp.save_npz(self.adj_matrix_file, adj_matrix)
        adj_matrix = sp.load_npz(self.adj_matrix_file)
        print(f'Done!')
        return adj_matrix

    def apply_filter(self, fmri_data, deg=15, lambda_min=0, lambda_max=2, tau=7, n_jobs=18):
        """Apply filter to fMRI data.

        Parameters
        ----------
        fmri_data : nibabel.Nifti1Image
            4D fMRI data
        deg : int, optional
            Degree of Chebyshev polynomial, by default 15
        lambda_min : float, optional
            Minimum eigenvalue, by default 0
        lambda_max : float, optional
            Maximum eigenvalue, by default 2
        tau : int, optional
            Tunable parameter, by default 7
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 18

        Returns
        -------
        fmri_data_filt : nibabel.Nifti1Image
            Filtered fMRI data
        """
        c = get_chebyshev_coeff(lambda l: np.exp(-tau*l), deg, lambda_min, lambda_max)
        L = sp.csgraph.laplacian(self.adj_matrix, normed=True).tocsc()
        L.data = np.nan_to_num(L.data)
        L.data[L.data == np.inf] = 0
        affine = fmri_data.affine
        print('Loading fMRI data...')
        fmri_data = fmri_data.get_fdata()
        print('Done!')
        filter_total = np.zeros(fmri_data.shape)
        wm_mask_data = self.wm_mask.astype(bool)
        def process_time(time):
            fmri_time_slice = fmri_data[:,:,:,time]
            f  = fmri_time_slice[wm_mask_data]

            # Apply filter
            f_filt = apply_filter(f, L, c, lambda_min=lambda_min, lambda_max=lambda_max)
            f_filt = f_filt.flatten()

            filt_wm = np.zeros(self.wm_mask.shape)
            filt_wm[self.linear_to_subscript_index[:,0], self.linear_to_subscript_index[:,1], self.linear_to_subscript_index[:,2]] = f_filt

            return filt_wm

        results = Parallel(n_jobs=n_jobs)(delayed(process_time)(time) for time in tqdm(range(fmri_data.shape[-1])))

        for time, result in tqdm(enumerate(results), total=len(results)):
            filter_total[:,:,:,time] = result

        fmri_data_filt = nib.Nifti1Image(filter_total, affine)

        return fmri_data_filt
    
class VSSFilter():
    def __init__(self, vasculature_peaks, wm_mask, adj_matrix_file, n=5, alpha=0.9, beta=50, threshold=None):
        """Class to create and apply vasculature-informed spatial smoothing filter for fMRI.

        Parameters
        ----------
        vasculature_peaks : nibabel.Nifti1Image
            Peak vasculature directions from SWI, stored in r x c x d x 3. Should be unit vectors.
        wm_mask : nibabel.Nifti1Image
            White matter mask to define voxels.
        adj_matrix_file : str
            Path to adjacency matrix .npz. Can be saved or path to save to.
        n : int, optional
            Number of connected neighbors, by default 5
        alpha : float, optional
            alpha parameter for sigmoid function, by default 0.9
        beta : int, optional
            beta parameter for sigmoid function, by default 50
        threshold : float, optional
            Threshold to apply to adjacency matrix, by default None

        Attributes
        ----------
        self.adj_matrix : scipy.sparse.csc.csc_matrix
            CSC sparse coherence-weighted adjacency matrix 
        self.unweighted_adj_matrix : scipy.sparse.csc.csc_matrix
            CSC sparse unweighted adjacency matrix
        self.subscript_to_linear_index : np.ndarray
            Mapping from voxel subscript to linear index. Linear index at x,y,z is subscript_to_linear_index[x,y,z]
        self.linear_to_subscript_index : np.ndarray
            Mapping from linear index to voxel subscript. Subscript at linear index i is linear_to_subscript_index[i]

        Methods
        -------
        apply_filter(fmri_data, deg=15, lambda_min=0, lambda_max=2, tau=7, n_jobs=18)
            Apply filter to fMRI data.
        -------
        """
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.adj_matrix_file = adj_matrix_file
        self.threshold = threshold

        # Load odfs and mask
        peaks = nib.funcs.as_closest_canonical(vasculature_peaks)
        wm_mask = nib.funcs.as_closest_canonical(wm_mask)
        self.peaks = peaks.get_fdata()
        self.wm_mask = wm_mask.get_fdata()

        # Sample peaks within mask
        self.peaks = self.peaks.reshape(-1, 3)
        self.n_voxels = self.wm_mask.sum().astype(int)

        # Create self.subscript_to_linear_index and self.linear_to_subscript_index
        self.subscript_to_linear_index = np.zeros(self.wm_mask.shape, dtype=int)
        self.subscript_to_linear_index[self.wm_mask > 0] = np.arange(self.n_voxels)
        self.linear_to_subscript_index = np.argwhere(self.wm_mask > 0)

    @cached_property
    def adj_matrix(self):
        # Check if adj_matrix_file exists
        if os.path.exists(self.adj_matrix_file):
            print(f'Loading {self.adj_matrix_file}...')
            adj_matrix = sp.load_npz(self.adj_matrix_file)
            print('Done!')
            return adj_matrix

        # Create adjacency matrix
        print(f'{self.adj_matrix_file} not found. Creating adjacency matrix...')

        # Get flattened voxels from mask
        voxels = np.argwhere(self.wm_mask > 0)
        voxels_indices = np.ravel_multi_index((voxels[:,0], voxels[:,1], voxels[:,2]), self.wm_mask.shape)
        X, Y, Z = np.meshgrid(
            np.arange(self.wm_mask.shape[0]), 
            np.arange(self.wm_mask.shape[1]), 
            np.arange(self.wm_mask.shape[2]),
        )

        # Get peaks in mask
        peaks = self.peaks
        peaks_nonzero = np.linalg.norm(peaks, axis=1) > 1e-10
        peaks[peaks_nonzero] = peaks[peaks_nonzero] / np.linalg.norm(peaks[peaks_nonzero], axis=1)[:, np.newaxis]

        # Get neighbors
        neighbors = get_neighbors(self.n)
        norm_dirs = neighbors / np.linalg.norm(neighbors, axis=1)[:, np.newaxis]

        # Repeat voxels to match number of neighbors
        voxels_repeated = np.repeat(voxels, len(neighbors), axis=0)

        # Tile neighbors for each voxel
        neighbors_repeated = np.tile(neighbors, (len(voxels), 1))
        neighbors = neighbors_repeated + voxels_repeated
        norm_dirs_repeated = np.tile(norm_dirs, (len(voxels),1)) 

        # Keep neighbors/voxels within bounds of image
        valid_indices = np.all(neighbors >= 0, axis=1) & np.all(neighbors < self.wm_mask.shape, axis=1)
        neighbors = neighbors[valid_indices]
        norm_dirs_repeated = norm_dirs_repeated[valid_indices, :]
        voxels_repeated = voxels_repeated[valid_indices]

        # Keep neighbors/voxels within mask
        mask_values = self.wm_mask[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]
        valid_indices = mask_values > 0
        neighbors = neighbors[valid_indices]
        norm_dirs_repeated = norm_dirs_repeated[valid_indices, :]
        voxels_repeated = voxels_repeated[valid_indices]

        neighbors_indices = np.ravel_multi_index((neighbors[:,0], neighbors[:,1], neighbors[:,2]), self.wm_mask.shape)
        voxels_repeated_indices = np.ravel_multi_index((voxels_repeated[:,0], voxels_repeated[:,1], voxels_repeated[:,2]), self.wm_mask.shape)

        voxels_repeated_peaks = peaks[voxels_repeated_indices,:]
        neighbor_repeated_peaks = peaks[neighbors_indices,:]
        voxels_peaks = peaks[voxels_indices,:]

        # Project voxels_repeated_peaks and neighbor_repeated_peaks onto norm_dirs_repeated
        voxels_repeated_proj = np.sum(voxels_repeated_peaks * norm_dirs_repeated, axis=1)[:,np.newaxis]*norm_dirs_repeated
        neighbor_repeated_proj = np.sum(neighbor_repeated_peaks * norm_dirs_repeated, axis=1)[:,np.newaxis]*norm_dirs_repeated

        # Calculate weights
        weights = np.abs(np.sum(voxels_repeated_proj * neighbor_repeated_proj, axis=1))
        weights = tunable_sigmoid(weights, alpha=self.alpha, beta=self.beta)

        row_indices = self.subscript_to_linear_index.flatten()[voxels_repeated_indices]
        col_indices = self.subscript_to_linear_index.flatten()[neighbors_indices]

        # Create adjacency matrix
        adj_matrix = sp.csc_matrix((weights, (row_indices, col_indices)), shape=(self.n_voxels, self.n_voxels))       
        
        # Threshold values below 1e-10
        if self.threshold is not None:
            adj_matrix.data[adj_matrix.data < self.threshold] = 0
            adj_matrix.eliminate_zeros()

        sp.save_npz(self.adj_matrix_file, adj_matrix)
        adj_matrix = sp.load_npz(self.adj_matrix_file)
        print(f'Done!')

        return adj_matrix

    def apply_filter(self, fmri_data, deg=15, lambda_min=0, lambda_max=2, tau=7, n_jobs=18):
        """Apply filter to fMRI data.

        Parameters
        ----------
        fmri_data : nibabel.Nifti1Image
            4D fMRI data
        deg : int, optional
            Degree of Chebyshev polynomial, by default 15
        lambda_min : float, optional
            Minimum eigenvalue, by default 0
        lambda_max : float, optional
            Maximum eigenvalue, by default 2
        tau : int, optional
            Tunable parameter, by default 7
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 18

        Returns
        -------
        fmri_data_filt : nibabel.Nifti1Image
            Filtered fMRI data
        """
        c = get_chebyshev_coeff(lambda l: np.exp(-tau*l), deg, lambda_min, lambda_max)
        L = sp.csgraph.laplacian(self.adj_matrix, normed=True).tocsc()
        affine = fmri_data.affine
        print('Loading fMRI data...')
        fmri_data = fmri_data.get_fdata()
        print('Done!')
        filter_total = np.zeros(fmri_data.shape)
        wm_mask_data = self.wm_mask.astype(bool)
        def process_time(time):
            fmri_time_slice = fmri_data[:,:,:,time]
            f  = fmri_time_slice[wm_mask_data]

            # Apply filter
            f_filt = apply_filter(f, L, c, lambda_min=lambda_min, lambda_max=lambda_max)
            f_filt = f_filt.flatten()

            filt_wm = np.zeros(self.wm_mask.shape)
            filt_wm[self.linear_to_subscript_index[:,0], self.linear_to_subscript_index[:,1], self.linear_to_subscript_index[:,2]] = f_filt

            return filt_wm

        results = Parallel(n_jobs=n_jobs)(delayed(process_time)(time) for time in tqdm(range(fmri_data.shape[-1])))
    

        for time, result in tqdm(enumerate(results), total=len(results)):
            filter_total[:,:,:,time] = result

        fmri_data_filt = nib.Nifti1Image(filter_total, affine)
        fmri_data_filt = nib.funcs.as_closest_canonical(fmri_data_filt)

        return fmri_data_filt