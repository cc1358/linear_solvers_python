import numpy as np
import scipy.sparse

def generate_dense_spd_matrix(n, max_cond=1e3, seed=None):
    """Generates a dense Symmetric Positive Definite matrix."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    A = np.dot(A, A.transpose()) # Make it symmetric positive semi-definite
    # Add a small multiple of identity to ensure positive definiteness and control condition number
    A += np.eye(n) * (1.0 / max_cond) * np.trace(A) / n # Heuristic for conditioning
    
    # Ensure it's truly SPD and try to control condition number better
    U, _, Vh = np.linalg.svd(A)
    s = np.linspace(1, max_cond, n) # Singular values from 1 to max_cond
    s = s / s[-1] # Normalize to have max singular value 1
    s = 1.0 + (s * (max_cond -1)) # Shift to desired range
    s = s[::-1] # Make largest singular value correspond to smallest index for linspace if needed
    
    A_new = U @ np.diag(s) @ Vh
    # Symmetrize A_new = (A_new + A_new.T)/2
    A_new = (A_new + A_new.T) / 2.0

    # Verify conditioning approximately
    # cond_A = np.linalg.cond(A_new)
    # print(f"Generated SPD matrix with n={n}, desired_max_cond~{max_cond}, actual_cond~{cond_A:.2e}")
    return A_new

def generate_dense_general_matrix(n, singular=False, seed=None, cond_target=None):
    """Generates a general dense matrix, possibly ill-conditioned or singular."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    if singular:
        A[:, -1] = A[:, 0] # Make last column dependent on first
    elif cond_target is not None:
        U, _, Vh = np.linalg.svd(A)
        s = np.logspace(0, np.log10(cond_target), n) # Singular values spread logarithmically up to cond_target
        s = s / s[-1] # Normalize max to 1
        s = 1.0 + s * (cond_target - 1) # Shift
        s = s[::-1] # Descending order
        A = U @ np.diag(s) @ Vh
    return A

def generate_tridiagonal_system(n, main_val=4.0, off_val=-1.0, seed=None):
    """Generates diagonals for a tridiagonal system and a random b vector."""
    rng = np.random.default_rng(seed)
    m_diag = np.full(n, main_val, dtype=np.double)
    l_diag = np.full(n - 1, off_val, dtype=np.double)
    u_diag = np.full(n - 1, off_val, dtype=np.double)
    b = rng.random(n)
    # Construct full matrix for reference if needed by some solvers or for verification
    A_full = np.diag(m_diag) + np.diag(l_diag, k=-1) + np.diag(u_diag, k=1)
    return l_diag, m_diag, u_diag, b, A_full

def generate_poisson_1d_problem(N_intervals, problem_type="sin_pi_x"):
    """
    Generates f_rhs and exact solution for -u'' = f on [0,1] with u(0)=u(1)=0.
    Args:
        N_intervals (int): Number of intervals. Grid has N_intervals+1 points.
        problem_type (str): Defines f(x) and exact solution.
                           "sin_pi_x": f(x) = sin(pi*x)
                           "const_one": f(x) = 1
    Returns:
        f_rhs (np.ndarray): RHS vector at grid points.
        h (float): Grid spacing.
        exact_u_func (callable): Function exact_u(x_pts) -> solution values.
        x_pts (np.ndarray): Grid points.
    """
    h = 1.0 / N_intervals
    x_pts = np.linspace(0, 1, N_intervals + 1)
    f_rhs = np.zeros(N_intervals + 1)
    exact_u_func = None

    if problem_type == "sin_pi_x":
        f_rhs = np.sin(np.pi * x_pts) * (np.pi**2) # If -u'' = pi^2 sin(pi x), u = sin(pi x)
                                                    # If you want f = sin(pi x), then u = sin(pi x)/pi^2
        f_rhs = np.sin(np.pi * x_pts) # Let -u'' = sin(pi*x)
        def u_exact(x): return np.sin(np.pi * x) / (np.pi**2)
        exact_u_func = u_exact
    elif problem_type == "const_one": # -u'' = 1 => u = -x^2/2 + cx + d. u(0)=0=>d=0. u(1)=0=>-1/2+c=0=>c=1/2. u = x/2 * (1-x)
        f_rhs = np.ones(N_intervals + 1)
        def u_exact(x): return x * (1 - x) / 2.0
        exact_u_func = u_exact
    
    # Ensure f_rhs is 0 at boundaries if u is fixed there (helps some formulations)
    # f_rhs[0] = 0; f_rhs[-1] = 0; # No, f_rhs is the source term, it can be non-zero at boundary
    return f_rhs, h, exact_u_func, x_pts


def generate_poisson_1d_periodic_problem(N_points, domain_L=2*np.pi, problem_type="cos_x"):
    """
    Generates f_rhs and exact solution for -u'' = f on [0,L] with periodic BCs.
    Solution u will be made to have mean zero.
    Args:
        N_points (int): Number of grid points.
        problem_type (str): Defines f(x) and exact solution.
                           "cos_x": f(x) = cos(x) => u(x) = cos(x)
    Returns:
        f_rhs (np.ndarray): RHS vector at grid points.
        h (float): Grid spacing.
        exact_u_func (callable): Function exact_u(x_pts) -> solution values (mean zero).
        x_pts (np.ndarray): Grid points.
    """
    h = domain_L / N_points
    x_pts = np.linspace(0, domain_L, N_points, endpoint=False)
    f_rhs = np.zeros(N_points)
    exact_u_func = None

    if problem_type == "cos_x": # -u'' = cos(x) => u = cos(x) (eigenvalue is 1)
        f_rhs = np.cos(x_pts) 
        def u_exact(x): 
            sol = np.cos(x)
            return sol - np.mean(sol) # Ensure mean zero
        exact_u_func = u_exact
    elif problem_type == "sin_2x": # -u'' = 4sin(2x) => u = sin(2x)
        f_rhs = 4 * np.sin(2 * x_pts)
        def u_exact(x):
            sol = np.sin(2*x)
            return sol - np.mean(sol)
        exact_u_func = u_exact

    # Ensure sum(f_rhs) is (close to) zero for periodic problem solvability
    f_rhs_mean = np.mean(f_rhs)
    if abs(f_rhs_mean) > 1e-9: # If f_rhs doesn't have mean zero
        #print(f"Adjusting f_rhs for periodic problem to have mean zero (original mean: {f_rhs_mean:.2e})")
        f_rhs -= f_rhs_mean
        
    return f_rhs, h, exact_u_func, x_pts
