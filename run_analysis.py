import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import solvers 
import matrix_generator

def run_single_test(solver_func, A, b, solver_params=None, problem_tag=""):
    print(f"\n--- Testing {solver_func.__name__} on {problem_tag} ---")
    if solver_params is None:
        solver_params = {}
    
    # Special handling for Thomas algorithm input
    if solver_func == solvers.solve_thomas:
        # Assume A contains the diagonals if it's passed this way for Thomas
        # This part needs to be structured based on how you pass tridiagonal systems
        # For now, let's assume 'solver_params' for Thomas will contain the diagonals
        if 'l_diag' not in solver_params or 'm_diag' not in solver_params or 'u_diag' not in solver_params:
            print("Error: Thomas solver requires l_diag, m_diag, u_diag in solver_params.")
            return None
        result = solver_func(solver_params['l_diag'], solver_params['m_diag'], solver_params['u_diag'], b)
    else: # General Ax=b solvers
        result = solver_func(A, b, **solver_params)

    # Print basic results
    print(f"  Method: {result.get('method_name', 'N/A')}")
    print(f"  Converged: {result.get('converged', 'N/A')}")
    print(f"  Time (s): {result.get('time_seconds', 0):.2e}")
    print(f"  Iterations: {result.get('iterations', 'N/A')}")
    print(f"  Residual Norm: {result.get('residual_norm', np.nan):.2e}")
    if result.get('svd_rank') is not None:
        print(f"  SVD Rank: {result['svd_rank']}")
    
    if "residual_history" in result and result["residual_history"] is not None:
        if len(result["residual_history"]) > 0:
            print(f"  Initial Residual: {result['residual_history'][0]:.2e} -> Final: {result['residual_history'][-1]:.2e}")
    return result

def plot_residual_histories(results_list, title="Convergence Plot"):
    plt.figure(figsize=(10, 6))
    for result in results_list:
        if result and result.get("residual_history") and result.get("converged"):
            history = result["residual_history"]
            # Ensure history is not empty and contains numbers
            if history and all(isinstance(item, (int, float)) for item in history):
                 # Limit iterations plotted if too many (e.g., for SOR on hard problems)
                max_iters_to_plot = 500
                iters_to_plot = min(len(history), max_iters_to_plot)
                label = f"{result['method_name']} ({result['iterations']} iters)"
                plt.semilogy(np.arange(1, iters_to_plot + 1), history[:iters_to_plot], marker='o', markersize=3, linestyle='-', label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Relative Residual Norm")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

def main_analysis():
    all_results_summary = []

    # --- Test Case 1: Small Dense SPD System ---
    print("======= TEST CASE 1: Small Dense SPD System =======")
    n_spd = 10
    A_spd = matrix_generator.generate_dense_spd_matrix(n_spd, max_cond=100, seed=0)
    b_spd = np.random.rand(n_spd)
    x_exact_spd = np.linalg.solve(A_spd, b_spd) # Reference

    test_solvers_std = [
        (solvers.solve_numpy_direct, {}),
        (solvers.solve_qr, {}),
        (solvers.solve_cholesky, {}),
        (solvers.solve_svd, {"rcond_svd": 1e-12}),
        (solvers.solve_gmres, {"rtol": 1e-7, "store_residual_history": True}),
        (solvers.solve_bicgstab, {"rtol": 1e-7, "store_residual_history": True}),
        (solvers.solve_sor, {"omega": 1.2, "rtol": 1e-7, "max_iter": 3000, "store_residual_history": True}),
    ]
    
    current_test_results = []
    for solver_func, params in test_solvers_std:
        result = run_single_test(solver_func, A_spd, b_spd, params, f"SPD {n_spd}x{n_spd}")
        if result and result.get("solution") is not None and result.get("converged"):
             result["L2_error_vs_exact"] = np.linalg.norm(result["solution"] - x_exact_spd)
        else:
             result["L2_error_vs_exact"] = np.nan
        all_results_summary.append(result)
        current_test_results.append(result)
    plot_residual_histories(current_test_results, title=f"Convergence on SPD {n_spd}x{n_spd} System")


    # --- Test Case 2: Tridiagonal System ---
    print("\n======= TEST CASE 2: Tridiagonal System =======")
    n_tri = 20
    l_diag, m_diag, u_diag, b_tri, A_tri_full = matrix_generator.generate_tridiagonal_system(n_tri, seed=1)
    x_exact_tri = np.linalg.solve(A_tri_full, b_tri)

    thomas_params = {'l_diag': l_diag, 'm_diag': m_diag, 'u_diag': u_diag}
    result_thomas = run_single_test(solvers.solve_thomas, A_tri_full, b_tri, thomas_params, f"Tridiagonal {n_tri}x{n_tri}")
    if result_thomas and result_thomas.get("solution") is not None and result_thomas.get("converged"):
        result_thomas["L2_error_vs_exact"] = np.linalg.norm(result_thomas["solution"] - x_exact_tri)
        # Recalculate residual for Thomas using full A if it was NaN
        result_thomas["residual_norm"] = solvers._calculate_relative_residual(A_tri_full, result_thomas["solution"], b_tri)

    all_results_summary.append(result_thomas)
    
    # Can also solve with general solvers for comparison
    current_test_results_tri = [result_thomas]
    for solver_func, params in test_solvers_std:
        if solver_func not in [solvers.solve_cholesky]: # Cholesky needs SPD, tri might not be perfectly so due to generation
            # SOR needs A_tri_full
            result = run_single_test(solver_func, A_tri_full, b_tri, params, f"Tridiagonal {n_tri}x{n_tri} (General Solver)")
            if result and result.get("solution") is not None and result.get("converged"):
                 result["L2_error_vs_exact"] = np.linalg.norm(result["solution"] - x_exact_tri)
            else:
                 result["L2_error_vs_exact"] = np.nan
            all_results_summary.append(result)
            current_test_results_tri.append(result)
    plot_residual_histories(current_test_results_tri, title=f"Convergence on Tridiagonal {n_tri}x{n_tri} System")


    # --- Test Case 3: 1D Poisson with Multigrid ---
    print("\n======= TEST CASE 3: 1D Poisson (Dirichlet BCs) with Multigrid =======")
    N_mg = 64 # Number of intervals
    f_rhs_mg, h_mg, exact_u_func_mg, x_pts_mg = matrix_generator.generate_poisson_1d_problem(N_mg, "sin_pi_x")
    
    mg_params = {"nu1": 2, "nu2": 2, "max_cycles": 30, "rtol": 1e-7, "exact_u_func": exact_u_func_mg}
    result_mg = solvers.solve_multigrid_two_grid_1d_poisson(f_rhs_mg, N_mg, **mg_params)
    
    print(f"  Method: {result_mg.get('method_name', 'N/A')}")
    print(f"  Converged: {result_mg.get('converged', 'N/A')}")
    print(f"  Time (s): {result_mg.get('time_seconds', 0):.2e}")
    print(f"  Cycles: {result_mg.get('iterations', 'N/A')}") # Iterations are cycles for MG
    print(f"  Error vs Exact: {result_mg.get('residual_norm', np.nan):.2e}") # residual_norm stores L2 error for this
    all_results_summary.append(result_mg)
    
    if result_mg.get("residual_history"):
        plt.figure(figsize=(8,5))
        plt.semilogy(np.arange(1, len(result_mg["residual_history"]) + 1), result_mg["residual_history"], marker='s', label=result_mg["method_name"])
        plt.xlabel("V-Cycle")
        plt.ylabel("Relative Residual Norm (of defect)")
        plt.title(f"Multigrid Convergence for 1D Poisson (N={N_mg})")
        plt.legend(); plt.grid(True,which="both",ls="--"); plt.show()

    # --- Test Case 4: 1D Periodic Poisson with FFT ---
    print("\n======= TEST CASE 4: 1D Poisson (Periodic BCs) with FFT =======")
    N_fft = 128 # Number of points
    f_rhs_fft, h_fft, exact_u_func_fft, x_pts_fft = matrix_generator.generate_poisson_1d_periodic_problem(N_fft, problem_type="cos_x")
    
    fft_params = {"exact_u_func": exact_u_func_fft, "domain_length": 2*np.pi}
    result_fft = solvers.solve_poisson_fft_1d_periodic(f_rhs_fft, **fft_params)

    print(f"  Method: {result_fft.get('method_name', 'N/A')}")
    print(f"  Time (s): {result_fft.get('time_seconds', 0):.2e}")
    print(f"  Error vs Exact (mean 0): {result_fft.get('residual_norm', np.nan):.2e}") # residual_norm stores L2 error
    all_results_summary.append(result_fft)


    # --- Summary Table (using Pandas) ---
    # Filter out None results if any solver failed catastrophically before returning a dict
    valid_results = [r for r in all_results_summary if r is not None]
    df_summary = pd.DataFrame(valid_results)
    
    # Select and reorder columns for better readability
    cols_to_show = [
        "method_name", "problem_tag", "converged", "time_seconds", 
        "iterations", "residual_norm", "L2_error_vs_exact", "problem_N", "svd_rank"
    ]
    # Filter existing columns
    existing_cols = [col for col in cols_to_show if col in df_summary.columns]
    
    print("\n\n======= Overall Summary Table =======")
    if not df_summary.empty:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df_summary[existing_cols])
    else:
        print("No results to display in summary.")

if __name__ == '__main__':
    main_analysis()
