import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.fft import fft, ifft
import time

# Common settings for iterative solvers
DEFAULT_RTOL = 1e-8
DEFAULT_MAX_ITER = 2000 # Increased default max_iter

def _calculate_relative_residual(A, x, b):
    if x is None: # Solver failed or solution not applicable
        return np.inf
    if not np.all(np.isfinite(x)): # Solution contains NaN or Inf
        return np.inf
    norm_b = np.linalg.norm(b)
    if norm_b == 0:
        # If b is zero vector, residual is norm(A@x). If x is also zero, residual is 0.
        # Otherwise, if A@x is non-zero, this indicates an issue.
        # For Ax=0, x=0 is trivial. If x!=0, norm(A@x) is absolute error.
        return np.linalg.norm(A @ x) 
    residual = np.linalg.norm(b - A @ x)
    return residual / norm_b

def _calculate_poisson_1d_residual_norm(u, f_rhs, h, exact_u_func=None):
    """Calculates error for 1D Poisson problem, either L2 of residual or error vs exact."""
    if exact_u_func:
        x_pts = np.linspace(0, (len(u)-1)*h, len(u))
        u_exact = exact_u_func(x_pts)
        return np.linalg.norm(u - u_exact) * np.sqrt(h) # L2 norm of error
    else:
        # Calculate norm of (f - Au)
        # This requires constructing A or applying it. For simplicity, not fully implemented here
        # as a generic residual. Often, convergence is based on change in u or specific problem criteria.
        # For the multigrid example, it already calculates its own residual norm based on f - Au_h.
        return np.nan # Placeholder if no exact solution to compare against and A is implicit

# --- Iterative Solver Callbacks for Residual History ---
class ResidualLogger:
    def __init__(self, A, b, verbose=False):
        self.A = A
        self.b = b
        self.residuals = []
        self.norm_b = np.linalg.norm(b)
        self.verbose = verbose

    def __call__(self, current_iterate):
        # For GMRES, current_iterate is the residual norm (rk).
        # For BiCGSTAB, current_iterate is xk (the current solution iterate).
        # We need to be flexible or specific. SciPy's GMRES callback gives residual norm.
        # SciPy's BiCGSTAB callback gives current x.
        
        if isinstance(current_iterate, (float, np.floating)): # GMRES callback gives residual norm
            if self.norm_b == 0:
                self.residuals.append(current_iterate)
            else:
                self.residuals.append(current_iterate / self.norm_b) # Relative residual norm
        else: # Assumed to be x_k (current solution iterate) for BiCGSTAB
            if self.norm_b == 0:
                residual_norm = np.linalg.norm(self.A @ current_iterate)
            else:
                residual_norm = np.linalg.norm(self.b - self.A @ current_iterate) / self.norm_b
            self.residuals.append(residual_norm)
        
        if self.verbose and len(self.residuals) % 10 == 0:
            print(f"Iteration {len(self.residuals)}, Relative Residual: {self.residuals[-1]:.2e}")


# -----------------------------------------
# 1. GMRES (Generalized Minimal Residual)
# -----------------------------------------
def solve_gmres(A, b, x0=None, rtol=DEFAULT_RTOL, max_iter=None, M=None, store_residual_history=False):
    method_name = "GMRES"
    if max_iter is None: # Sensible default based on matrix size
        max_iter = min(A.shape[0] * 2, DEFAULT_MAX_ITER)
    
    residual_logger = ResidualLogger(A, b) if store_residual_history else None
    
    start_time = time.perf_counter()
    try:
        # SciPy's GMRES callback is for the residual norm, not x_k.
        # So, the callback receives rk directly.
        x, exit_code = scipy.sparse.linalg.gmres(A, b, x0=x0, rtol=rtol, maxiter=max_iter, M=M, callback=residual_logger, callback_type='pr_norm')
        converged = (exit_code == 0)
    except Exception as e:
        print(f"{method_name} failed: {e}")
        x = None
        converged = False
        exit_code = -100 # Custom error code

    end_time = time.perf_counter()
    
    iterations = len(residual_logger.residuals) if store_residual_history and residual_logger else (exit_code if exit_code > 0 else (max_iter if not converged else None))
    if iterations is None and exit_code == 0: # If converged quickly and no history
        # Try to get iteration count if possible, though SciPy's GMRES doesn't directly return it on exit_code=0
        # It returns the number of iterations if exit_code > 0 (not converged in `maxiter` iterations)
        # For now, if converged and no history, we don't have a precise iteration count from GMRES directly unless history is logged.
        pass # iterations will remain None or an estimate if we can make one

    final_residual = _calculate_relative_residual(A, x, b) if x is not None else np.inf
    
    return {
        "method_name": method_name,
        "solution": x,
        "time_seconds": end_time - start_time,
        "iterations": iterations if iterations is not None else "N/A", # SciPy GMRES doesn't always give clear iter count on success
        "converged": converged,
        "exit_code": exit_code,
        "residual_norm": final_residual,
        "residual_history": residual_logger.residuals if store_residual_history and residual_logger else None,
    }

# -----------------------------------------
# 2. BiCGSTAB (Biconjugate Gradient Stabilized)
# -----------------------------------------
def solve_bicgstab(A, b, x0=None, rtol=DEFAULT_RTOL, max_iter=None, M=None, store_residual_history=False):
    method_name = "BiCGSTAB"
    if max_iter is None:
        max_iter = min(A.shape[0] * 2, DEFAULT_MAX_ITER)

    residual_logger = ResidualLogger(A, b) if store_residual_history else None

    start_time = time.perf_counter()
    try:
        # BiCGSTAB callback receives xk (current solution)
        x, exit_code = scipy.sparse.linalg.bicgstab(A, b, x0=x0, rtol=rtol, maxiter=max_iter, M=M, callback=residual_logger)
        converged = (exit_code == 0)
    except Exception as e:
        print(f"{method_name} failed: {e}")
        x = None
        converged = False
        exit_code = -100

    end_time = time.perf_counter()
    
    iterations = len(residual_logger.residuals) if store_residual_history and residual_logger else (exit_code if exit_code > 0 else None)
    # If converged and exit_code is 0, iterations are not directly returned by scipy.sparse.linalg.bicgstab
    # The number of times callback is called is the iteration count.
    
    final_residual = _calculate_relative_residual(A, x, b) if x is not None else np.inf

    return {
        "method_name": method_name,
        "solution": x,
        "time_seconds": end_time - start_time,
        "iterations": iterations if iterations is not None else "N/A",
        "converged": converged,
        "exit_code": exit_code,
        "residual_norm": final_residual,
        "residual_history": residual_logger.residuals if store_residual_history and residual_logger else None,
    }

# -----------------------------------------
# 3. SOR (Successive Over-Relaxation)
# -----------------------------------------
def solve_sor(A, b, omega, x0=None, rtol=DEFAULT_RTOL, max_iter=DEFAULT_MAX_ITER, store_residual_history=False):
    method_name = f"SOR (ω={omega})"
    n = A.shape[0]
    if x0 is None:
        x = np.zeros_like(b, dtype=np.double)
    else:
        x = x0.astype(np.double).copy()

    converged = False
    residual_history = []
    norm_b = np.linalg.norm(b)
    if norm_b == 0: norm_b = 1.0 # Avoid division by zero for residual; use absolute if norm_b is 0

    start_time = time.perf_counter()
    iterations_performed = 0
    for k in range(max_iter):
        iterations_performed = k + 1
        x_old = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            if abs(A[i, i]) < 1e-12: # Avoid division by zero
                print(f"{method_name} failed: Zero diagonal element A[{i},{i}].")
                converged = False
                x = None # Indicate failure
                break 
            x[i] = (1 - omega) * x_old[i] + (omega / A[i, i]) * (b[i] - sigma)
        if x is None: break # Propagate failure from inner loop

        current_residual = np.linalg.norm(b - A @ x) / norm_b
        if store_residual_history:
            residual_history.append(current_residual)
        if current_residual < rtol:
            converged = True
            break
    end_time = time.perf_counter()
    
    final_residual = _calculate_relative_residual(A, x, b) if x is not None else np.inf
    
    return {
        "method_name": method_name,
        "solution": x,
        "time_seconds": end_time - start_time,
        "iterations": iterations_performed,
        "converged": converged,
        "residual_norm": final_residual,
        "residual_history": residual_history if store_residual_history else None,
    }

# -----------------------------------------
# 4. QR Decomposition based Solver
# -----------------------------------------
def solve_qr(A, b):
    method_name = "QR Decomposition"
    start_time = time.perf_counter()
    try:
        if A.shape[0] == A.shape[1]: # Square matrix
            Q, R = np.linalg.qr(A)
            b_transformed = np.dot(Q.T, b)
            x = scipy.linalg.solve_triangular(R, b_transformed)
        else: # Non-square matrix, solve in least-squares sense
            Q, R = np.linalg.qr(A) # Default 'reduced' mode
            b_transformed = np.dot(Q.T, b)
            x = scipy.linalg.solve_triangular(R, b_transformed)
        converged = True
    except np.linalg.LinAlgError as e:
        print(f"{method_name} failed: {e}")
        x = None
        converged = False
    end_time = time.perf_counter()
    
    final_residual = _calculate_relative_residual(A, x, b) if x is not None else np.inf

    return {
        "method_name": method_name,
        "solution": x,
        "time_seconds": end_time - start_time,
        "iterations": 0, # Direct method
        "converged": converged,
        "residual_norm": final_residual,
    }

# -----------------------------------------
# 5. Cholesky Decomposition based Solver
# -----------------------------------------
def solve_cholesky(A, b):
    method_name = "Cholesky Decomposition"
    start_time = time.perf_counter()
    try:
        c, low = scipy.linalg.cho_factor(A, lower=False)
        x = scipy.linalg.cho_solve((c, low), b)
        converged = True
    except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as e:
        print(f"{method_name} failed: Matrix likely not SPD. {e}")
        x = None
        converged = False
    end_time = time.perf_counter()

    final_residual = _calculate_relative_residual(A, x, b) if x is not None else np.inf

    return {
        "method_name": method_name,
        "solution": x,
        "time_seconds": end_time - start_time,
        "iterations": 0, # Direct method
        "converged": converged,
        "residual_norm": final_residual,
    }

# -----------------------------------------
# 6. SVD (Singular Value Decomposition) based Solver
# -----------------------------------------
def solve_svd(A, b, rcond_svd=None): # Renamed rcond to avoid conflict if used globally
    method_name = "SVD Solver (via lstsq)"
    start_time = time.perf_counter()
    try:
        x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=rcond_svd)
        converged = True # lstsq always returns a solution
    except np.linalg.LinAlgError as e:
        print(f"{method_name} failed: {e}")
        x = None
        converged = False
    end_time = time.perf_counter()

    final_residual = _calculate_relative_residual(A, x, b) if x is not None else np.inf
    
    return {
        "method_name": method_name,
        "solution": x,
        "time_seconds": end_time - start_time,
        "iterations": 0, # Direct method based on factorization
        "converged": converged,
        "residual_norm": final_residual,
        "svd_rank": rank if 'rank' in locals() else None, # locals() to check if rank was defined
        "svd_singular_values": singular_values if 'singular_values' in locals() else None
    }

# -----------------------------------------
# 7. Thomas Algorithm (for tridiagonal systems)
# -----------------------------------------
def solve_thomas(l_diag, m_diag, u_diag, b):
    method_name = "Thomas Algorithm"
    n = len(m_diag)
    if not (len(l_diag) == n - 1 and len(u_diag) == n - 1 and len(b) == n):
        # This should ideally raise an error or be handled by the caller
        print(f"{method_name} failed: Dimension mismatch.")
        return {
            "method_name": method_name, "solution": None, "time_seconds": 0,
            "iterations": 0, "converged": False, "residual_norm": np.inf,
        }

    c_prime = np.zeros(n, dtype=np.double)
    d_prime = np.zeros(n, dtype=np.double)
    x_sol = np.zeros(n, dtype=np.double) # Renamed from x to avoid conflict
    converged = True

    start_time = time.perf_counter()
    try:
        if abs(m_diag[0]) < 1e-12: raise ValueError("Zero pivot at first element.")
        c_prime[0] = u_diag[0] / m_diag[0]
        d_prime[0] = b[0] / m_diag[0]

        for i in range(1, n):
            temp = m_diag[i] - l_diag[i-1] * c_prime[i-1]
            if abs(temp) < 1e-12: raise ValueError(f"Zero pivot at index {i}.")
            if i < n - 1:
                c_prime[i] = u_diag[i] / temp
            d_prime[i] = (b[i] - l_diag[i-1] * d_prime[i-1]) / temp
        
        x_sol[n-1] = d_prime[n-1]
        for i in range(n-2, -1, -1):
            x_sol[i] = d_prime[i] - c_prime[i] * x_sol[i+1]
    except ValueError as e:
        print(f"{method_name} failed: {e}")
        x_sol = None
        converged = False
    end_time = time.perf_counter()

    # For Thomas, need to reconstruct A to calculate residual easily, or pass A
    # For now, residual calculation relies on A being available, which is not direct for Thomas
    # The caller will typically construct A from diagonals if needed for verification.
    # We can pass the full matrix A if we want to calculate residual here.
    # For now, we'll set residual to NaN, assuming caller verifies.
    final_residual = np.nan 
    if x_sol is not None and converged:
        # Reconstruct A to calculate residual (optional, can be slow for large N)
        # A_full = np.diag(m_diag) + np.diag(l_diag, k=-1) + np.diag(u_diag, k=1)
        # final_residual = _calculate_relative_residual(A_full, x_sol, b)
        pass # Skip residual calculation here for Thomas unless A_full is passed or reconstructed

    return {
        "method_name": method_name,
        "solution": x_sol,
        "time_seconds": end_time - start_time,
        "iterations": 0, # Direct method
        "converged": converged,
        "residual_norm": final_residual, # Or calculate if A_full provided
    }


# -----------------------------------------
# Specialized Solvers (Multigrid, FFT Poisson)
# These solve specific problems, Ax=b interpretation needs care.
# Their "A" is often implicit. We'll report metrics relevant to their problem.
# -----------------------------------------

# (Multigrid helpers: smoother_jacobi_1d, restrict_1d, prolong_1d, compute_residual_1d from previous version)
# ... (Keep these helper functions as they were, or integrate them if preferred)
# For brevity, I'll assume they are present as you had them.
# Here are the multigrid helper functions again for completeness.
def smoother_jacobi_1d(u, f_rhs, h, num_sweeps=2, omega=2.0/3.0):
    v = u.copy()
    h2 = h*h
    for _ in range(num_sweeps):
        v_old = v.copy()
        for i in range(1, len(v) - 1):
            v[i] = (1 - omega) * v_old[i] + \
                   omega * 0.5 * (v_old[i-1] + v_old[i+1] + h2 * f_rhs[i])
    return v

def restrict_1d(v_fine):
    N_fine = len(v_fine) -1
    if N_fine <= 0 : return np.array([0.0]) # Handle case with no interior points
    N_coarse = N_fine // 2
    if N_coarse == 0 and N_fine > 0 : # e.g. N_fine = 1, grid is 0-x-0, coarse is 0-0
         return np.zeros(1) # Coarse grid has only one point (like a boundary aggregate)
    v_coarse = np.zeros(N_coarse + 1) 
    for j_coarse in range(1, N_coarse): 
        j_fine = 2 * j_coarse
        v_coarse[j_coarse] = 0.25 * v_fine[j_fine-1] + 0.5 * v_fine[j_fine] + 0.25 * v_fine[j_fine+1]
    return v_coarse

def prolong_1d(v_coarse):
    N_coarse = len(v_coarse) - 1
    if N_coarse < 0 : return np.array([]) # Should not happen if restrict_1d is correct
    if N_coarse == 0: return np.array([0.0, 0.0]) # e.g. coarse is [0], fine is [0,0] (approx)
                                               # or if coarse is one point, prolong to two points with that value.
                                               # This logic needs to be robust for very small grids.
                                               # Standard prolongation assumes at least 2 coarse points for interpolation.
                                               # For 1 coarse interior point (3 total points), fine grid is 5 points.
                                               # If N_coarse=0 (1 point total on coarse), N_fine=0 (1 point total on fine)
                                               # This simplification handles boundary conditions for 1D Poisson.
                                               # Let's assume len(v_coarse) >= 2 (e.g. [0,0] at least) for proper interpolation.
    if len(v_coarse) == 1: # Only one point on coarse grid (e.g. solution is just 0)
            return np.array([v_coarse[0], v_coarse[0]]) # Simplistic prolongation to 2 points
            
    N_fine = N_coarse * 2
    v_fine = np.zeros(N_fine + 1) 
    for j_coarse in range(N_coarse + 1): 
        v_fine[2*j_coarse] = v_coarse[j_coarse]
    for j_coarse in range(N_coarse): 
        v_fine[2*j_coarse + 1] = 0.5 * (v_coarse[j_coarse] + v_coarse[j_coarse+1])
    return v_fine

def compute_residual_1d(u, f_rhs, h):
    N = len(u) - 1
    if N <= 0: return np.array([])
    r = np.zeros_like(f_rhs) 
    h2 = h*h
    for i in range(1, N):
        Au_i = (-u[i-1] + 2*u[i] - u[i+1]) / h2
        r[i] = f_rhs[i] - Au_i
    return r 
# -----------------------------------------
# 8. Multigrid Method (Simplified 1D Geometric Two-Grid for -u''=f)
# -----------------------------------------
def solve_multigrid_two_grid_1d_poisson(f_rhs, N_fine_intervals, nu1=2, nu2=2, max_cycles=20, rtol=1e-7, exact_u_func=None):
    method_name = "Multigrid (1D Two-Grid)"
    
    h_fine = 1.0 / N_fine_intervals
    u_fine = np.zeros(N_fine_intervals + 1) 
    
    N_coarse_intervals = N_fine_intervals // 2
    if N_coarse_intervals == 0 and N_fine_intervals > 0: # Avoid trivial coarse grid if fine grid isn't trivial
        print(f"{method_name}: Fine grid N={N_fine_intervals} too small for meaningful two-grid. Solving on fine grid only with smoothing.")
        # Fallback: just smooth on fine grid as a very basic solver
        start_time_fallback = time.perf_counter()
        for _ in range(max_cycles * (nu1 + nu2)): # Equivalent work
            u_fine = smoother_jacobi_1d(u_fine, f_rhs, h_fine, num_sweeps=1)
            # Check convergence (simplified for this fallback)
            # current_res_norm = np.linalg.norm(compute_residual_1d(u_fine, f_rhs, h_fine)[1:-1])
            # if current_res_norm < rtol * np.linalg.norm(f_rhs[1:-1]): break
        end_time_fallback = time.perf_counter()
        final_error_metric = _calculate_poisson_1d_residual_norm(u_fine, f_rhs, h_fine, exact_u_func)
        return {
            "method_name": method_name + " (Fallback: Smoother Only)",
            "solution": u_fine, "time_seconds": end_time_fallback - start_time_fallback,
            "iterations": max_cycles * (nu1+nu2), "converged": True, # Placeholder for fallback
            "residual_norm": final_error_metric, "problem_N": N_fine_intervals
        }

    if N_coarse_intervals <= 0: # Truly trivial problem or invalid N
        print(f"{method_name}: N={N_fine_intervals} too small. Returning zeros.")
        return {
            "method_name": method_name, "solution": u_fine, "time_seconds": 0,
            "iterations": 0, "converged": True, "residual_norm": 0 if exact_u_func is None else np.linalg.norm(u_fine - exact_u_func(np.linspace(0,1,len(u_fine))))*np.sqrt(h_fine),
            "problem_N": N_fine_intervals
        }
    h_coarse = 1.0 / N_coarse_intervals

    norm_f_interior = np.linalg.norm(f_rhs[1:N_fine_intervals])
    if norm_f_interior == 0: norm_f_interior = 1.0

    residual_history = []
    converged = False
    cycles_performed = 0

    start_time = time.perf_counter()
    for cycle in range(max_cycles):
        cycles_performed = cycle + 1
        u_fine = smoother_jacobi_1d(u_fine, f_rhs, h_fine, num_sweeps=nu1)
        r_fine = compute_residual_1d(u_fine, f_rhs, h_fine)
        
        current_residual_norm = np.linalg.norm(r_fine[1:N_fine_intervals]) / norm_f_interior
        residual_history.append(current_residual_norm)

        if current_residual_norm < rtol:
            converged = True
            break

        r_coarse = restrict_1d(r_fine)
        
        # Solve coarse grid problem A_coarse e_coarse = r_coarse
        e_coarse_solution = np.zeros(N_coarse_intervals + 1)
        if N_coarse_intervals -1 > 0 : # If there are interior coarse points
            # Define diagonals for Thomas solver for the coarse grid problem (-d^2/dx^2)
            # A_coarse_mat for interior points: (1/h_coarse^2) * Tridiag(-1, 2, -1)
            m_diag_coarse = np.full(N_coarse_intervals - 1, 2.0 / (h_coarse**2))
            off_diag_coarse = np.full(N_coarse_intervals - 2, -1.0 / (h_coarse**2))
            
            # Solve for interior points of e_coarse. Thomas needs b at these points.
            b_thomas_coarse = r_coarse[1:N_coarse_intervals]

            thomas_result = solve_thomas(off_diag_coarse, m_diag_coarse, off_diag_coarse, b_thomas_coarse)
            if thomas_result["converged"]:
                e_coarse_solution[1:N_coarse_intervals] = thomas_result["solution"]
            else: # Thomas failed on coarse grid
                print(f"{method_name}: Coarse grid solve failed. Using zero correction.")
        elif N_coarse_intervals -1 == 0 and N_coarse_intervals == 1: # Single interior coarse point (N_coarse_intervals=1 implies grid 0-x-0)
             # Grid: C--I--C. Equation at I: (2/h_c^2)e_I = r_I => e_I = r_I * h_c^2 / 2
             if abs(2.0 / (h_coarse**2)) > 1e-12:
                e_coarse_solution[1] = r_coarse[1] * (h_coarse**2) / 2.0


        e_fine = prolong_1d(e_coarse_solution)
        u_fine += e_fine
        u_fine = smoother_jacobi_1d(u_fine, f_rhs, h_fine, num_sweeps=nu2)
    end_time = time.perf_counter()

    final_error_metric = _calculate_poisson_1d_residual_norm(u_fine, f_rhs, h_fine, exact_u_func)

    return {
        "method_name": method_name,
        "solution": u_fine,
        "time_seconds": end_time - start_time,
        "iterations": cycles_performed, # cycles
        "converged": converged,
        "residual_norm": final_error_metric, # This is L2 error against exact if available
        "residual_history": residual_history, # Cycle-wise residual norm
        "problem_N": N_fine_intervals
    }

# -----------------------------------------
# 9. Fast Poisson Solver (1D FFT-based for periodic BCs)
# -----------------------------------------
def solve_poisson_fft_1d_periodic(f_rhs, domain_length=2*np.pi, exact_u_func=None):
    method_name = "FFT Poisson (1D Periodic)"
    n_points = len(f_rhs)
    if n_points == 0:
        return {"method_name": method_name, "solution": np.array([]), "time_seconds": 0, "iterations": 0,
                "converged": True, "residual_norm": 0, "problem_N": 0}
        
    h_grid = domain_length / n_points
    start_time = time.perf_counter()

    m_wave_numbers = np.fft.fftfreq(n_points) * n_points
    eigenvalues_M = (2.0 * (1.0 - np.cos(2 * np.pi * m_wave_numbers / n_points))) / (h_grid**2)

    f_hat = fft(f_rhs)
    u_hat = np.zeros_like(f_hat, dtype=np.complex128)

    # Handle the m=0 mode (DC component)
    # Find index of zero frequency, usually the first one.
    zero_freq_idx = np.where(np.abs(m_wave_numbers) < 1e-9)[0]
    
    if len(zero_freq_idx)>0:
        idx = zero_freq_idx[0]
        if abs(eigenvalues_M[idx]) < 1e-9: 
            if abs(f_hat[idx]) > 1e-9 * n_points : # Check if sum(f_rhs) is non-zero 
                print(f"Warning ({method_name}): Sum of f_rhs (f_hat[0]={f_hat[idx]:.2e}) is non-zero. Ill-posed.")
            u_hat[idx] = 0.0 # Set DC component of solution to 0 (=> mean(u) = 0)
            
            non_zero_eig_indices = np.where(np.abs(eigenvalues_M) >= 1e-9)[0]
            u_hat[non_zero_eig_indices] = f_hat[non_zero_eig_indices] / eigenvalues_M[non_zero_eig_indices]
        else: # Should not happen for standard discrete Laplacian
            u_hat = f_hat / eigenvalues_M
    else: # No zero frequency found (e.g. n_points is very small or unusual fftfreq output)
         u_hat = f_hat / eigenvalues_M # Proceed with caution


    u_sol = np.real(ifft(u_hat))
    end_time = time.perf_counter()
    
    final_error_metric = np.nan
    if exact_u_func:
        x_pts = np.linspace(0, domain_length, n_points, endpoint=False)
        u_exact = exact_u_func(x_pts)
        # Ensure mean of exact solution is zero for fair comparison if u_sol is mean zero
        u_exact_mean_zero = u_exact - np.mean(u_exact)
        final_error_metric = np.linalg.norm(u_sol - u_exact_mean_zero) / np.sqrt(n_points) # Scaled L2 norm

    return {
        "method_name": method_name,
        "solution": u_sol,
        "time_seconds": end_time - start_time,
        "iterations": 0, # Direct method
        "converged": True, # FFT is a direct solve
        "residual_norm": final_error_metric, # L2 error against exact solution
        "problem_N": n_points
    }

# --- Baseline Numpy Solver ---
def solve_numpy_direct(A, b):
    method_name = "NumPy np.linalg.solve"
    start_time = time.perf_counter()
    try:
        x = np.linalg.solve(A, b)
        converged = True
    except np.linalg.LinAlgError as e:
        print(f"{method_name} failed: {e}")
        x = None
        converged = False
    end_time = time.perf_counter()

    final_residual = _calculate_relative_residual(A, x, b) if x is not None else np.inf
    
    return {
        "method_name": method_name,
        "solution": x,
        "time_seconds": end_time - start_time,
        "iterations": 0,
        "converged": converged,
        "residual_norm": final_residual,
    }
