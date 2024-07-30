import numpy as np
import scipy as sp
from tqdm import tqdm
import stripy
from itertools import product
from dipy.reconst.shm import real_sh_tournier, real_sh_descoteaux

def cart2sphere(x, y, z):
    """Convert Cartesian coordinates to spherical coordinates"""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x) + np.pi
    phi = np.arccos(z/r)
    return r, theta, phi

def sphere2cart(r, theta, phi):
    """Convert spherical coordinates to Cartesian coordinates"""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def get_chebyshev_coeff(k, deg=15, lambda_min=0, lambda_max=2):
    """Get Chebyshev coefficients for polynomial approximation of filter k

    Parameters
    ----------
    k : function handle
        filter kernel to act on eigenvalues
    deg : int
        degree of polynomial, by default 15
    lambda_min : int, optional
        minimum eigenvalue, by default 0
    lambda_max : int, optional
        maximum, by default 2

    Returns
    -------
    c : ndarray
        (deg+1,) array of Chebyshev coefficients
    """
    l = np.linspace(lambda_min, lambda_max, 100)
    l_shift = (l - lambda_min) / (lambda_max - lambda_min) * 2 - 1
    c = np.polynomial.chebyshev.chebfit(l_shift, k(l), deg)
    return c

def apply_filter(f, L, c, lambda_min=0, lambda_max=2):
    """Apply Chebyshev polynomial filter to signal f

    Parameters
    ----------
    f : ndarray
        flattened signal, should be same size as rows or cols of L
    L : sparse CSC array
        sparse normalized Laplacian array from adjacency matrix
    c : ndarray
        (deg+1,) array of Chebyshev coefficients
    lambda_min : int, optional
        minimum eigenvalue, by default 0
    lambda_max : int, optional
        maximum eigenvalue, by default 2
    
    Returns
    -------
    f_filt : ndarray
        filtered signal
    """
    a1 = (lambda_max-lambda_min)/2
    a2 = (lambda_max+lambda_min)/2
    deg = c.shape[0] - 1
    T_minus_2_f = f
    T_minus_1_f = (1/a1*L@f - a2/a1*f)
    f_filt = c[0]*T_minus_2_f + c[1]*T_minus_1_f
    for k in range(2, deg):
        T_n_f = 2/a1*(L@T_minus_1_f - a2*T_minus_1_f) - T_minus_2_f
        f_filt += c[k]*T_n_f
        T_minus_2_f = T_minus_1_f
        T_minus_1_f = T_n_f
    return f_filt

def get_filter(L, coefficients, lambda_min=0, lambda_max=2):
    """Return p(L), the polynomial approximation of the filter kernel k

    Parameters
    ----------
    L : sparse CSC array
        normalized Laplacian array of adjacency matrix
    coefficients : ndarray
        (deg+1,) array of Chebyshev coefficients
    lambda_min : float
        minimum eigenvalue, by default 0
    lambda_max : float
        maximum eigenvalue, by default 2

    Returns
    -------
    p_L : sparse CSC array
        polynomial approximation of the filter kernel k
    """
    a1 = (lambda_max - lambda_min) / 2
    a2 = (lambda_max + lambda_min) / 2
    deg = coefficients.shape[0]
    # L_shift = 1/a1*L - a2/a1*sp.sparse.eye(L.shape[0])

    # Initialize the Chebyshev polynomial values
    T_minus_2_L = sp.sparse.eye(L.shape[0]) # Identity matrix
    T_minus_1_L = 1/a1*L - a2/a1*sp.sparse.eye(L.shape[0])
    
    # Initialize the result matrix
    p_L = coefficients[0] * T_minus_2_L + coefficients[1] * T_minus_1_L
    
    # Evaluate the Chebyshev polynomial at the Laplacian matrix
    deg = len(coefficients)
    for k in tqdm(range(2, deg), total=deg-2):
        T_n_L = 2/a1*(L@T_minus_1_L - a2*T_minus_1_L) - T_minus_2_L
        p_L += coefficients[k] * T_n_L
        
        # Update T_minus_2_L and T_minus_1_L for the next iteration
        T_minus_2_L = T_minus_1_L.copy()
        T_minus_1_L = T_n_L.copy()
        
    return p_L

def tunable_sigmoid(x, alpha=0.9, beta=50):
    num = ((1-alpha)*x)**beta
    denom = num + ((1-x)*alpha)**beta
    if isinstance(x, np.ndarray):
        num[denom == 0] = 0
        denom[denom == 0] = 1
    return num / denom

def get_neighbors(n, remove_parallel=True):
    """Get neighbors in n x n x n cube, n=3 or n=5. Remove parallel vectors that aren't
    farthest away if remove_parallel=True.
    """
    if n == 3:
        rep_list = [-1, 0, 1]
        remove_list = []
    elif n == 5:
        rep_list = [-2, -1, 0, 1, 2]
        remove_list = [2, 3, np.sqrt(8), np.sqrt(12)]
    else:
        raise ValueError("n must be 3 or 5")

    neighbors = np.array(list(product(rep_list, repeat=3)))
    neighbors[:,[0,2]] = neighbors[:,[2,0]] # match matlab

    # Remove 0,0,0
    neighbors = neighbors[~np.all(neighbors == 0, axis=1)]

    if remove_parallel:
        normalized_neighbors = neighbors / np.linalg.norm(neighbors, axis=1)[:,np.newaxis]
        cos_theta = np.dot(normalized_neighbors, normalized_neighbors.T)
        parallel_neighbors = np.zeros(cos_theta.shape)
        parallel_neighbors[cos_theta > 0.99] = 1
        norms = np.linalg.norm(neighbors, axis=1)
        include = np.zeros(len(parallel_neighbors[0]))
        for i in range(len(parallel_neighbors[0])):
            include[i] = norms[i] <= np.min(norms[parallel_neighbors[:,i] == 1])
        neighbors = neighbors[include.astype(bool)]

    return neighbors

def create_sampling_template(n):
    """Create sampling template using icosahedron with equally distributed points

    Parameters
    ----------
    n : int
        3 or 5, number of connected neighbors for graph. Used to determine size of solid
        angle to sample.

    Returns
    -------
    template_directions : np.ndarray
        Vertices of icosahedron to sample on sphere. Directed around z-axis.
    """
    fmesh = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=5, include_face_points=False)
    neighbors = get_neighbors(n)
    solid_angle = 4*np.pi/len(neighbors)
    zdir = np.array([0.0, 0.0, 1.0])

    # Get vertices of fmesh
    vertices = fmesh.points

    # Find vertices in template
    cos_theta = 1 - solid_angle / (2 * np.pi)
    template_cos_theta = np.dot(vertices, zdir)
    template_directions = vertices[template_cos_theta > cos_theta,:]
    return template_directions

def sample_odfs(directions, template, odfs_sh, sh_method='tournier', sh_order=8):
    """Sample all ODFs using a template rotated to a set of specific directions

    Parameters
    ----------
    directions : (N,3) arraylike
        N unit vector directions to take samples at
    template : (M,3) arraylike
        template centered around z-axis to sample on sphere with M vertices
    odfs_sh : (45,K) arraylike
        Set of ODFs in spherical harmonics, expected to be length 45 for order 8
    sh_method : str, optional
        Method to calculate spherical harmonics, by default 'tournier'. (Or 'descoteaux')
    sh_order : int, optional
        Spherical harmonics order, by default 8

    Returns
    -------
    sample : (N,K) arraylike
        Sampled ODFs at each direction. Nth direction and Kth ODF are at sample [N,K]
    """
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 100)

    # Rotate template to desired direction
    zdir = np.array([0.0, 0.0, 1.0])
    template_pts = np.ones((template.shape[0]*directions.shape[0], 2))
    sample = np.zeros((directions.shape[0], odfs_sh.shape[1]))
    for i, direction in tqdm(enumerate(directions), total=directions.shape[0]):
        if np.all(direction == zdir):
            template_rotated = template
        else:
            v = np.cross(zdir, direction)
            s = np.linalg.norm(v)
            c = np.dot(zdir, direction)
            kmat = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
            if s**2 > 1e-10:
                R = np.eye(3) + kmat + kmat @ kmat * (1 - c) / (s ** 2)
            else:
                R = np.eye(3)
            template_rotated = R @ template.T
            template_rotated = template_rotated.T
        _, template_theta, template_phi = cart2sphere(template_rotated[:,0], template_rotated[:,1], template_rotated[:,2])
        template_theta = template_theta - np.pi
        if sh_method == 'tournier':
            shmat, _, _ = real_sh_tournier(sh_order, template_phi, template_theta)
        elif sh_method == 'descoteaux':
            shmat, _, _ = real_sh_descoteaux(sh_order, template_phi, template_theta)
        sample[i, :] = np.sum(shmat @ odfs_sh, axis=0)
    return sample/template.shape[0]