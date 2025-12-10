
import numpy as np
import scipy
from scipy.special import xlogy
# import sklearn
import sklearn.metrics
import sklearn.gaussian_process.kernels as kernels
import matplotlib.pyplot as plt

from . import utilities



def krr_laplacian_oneshot(training_spatial_locations, training_targets, kernel_obj, effective_Laplacian, lam_krr, omega_lap, kernel_mat=None):
    """
    Performs one-shot Kernel Ridge Regression with a Laplacian regularizer.

    This function solves the system of linear equations to find the optimal
    weights for a Kernel Ridge Regression model that incorporates a Laplacian
    regularizer.

    Args:
        training_spatial_locations (np.ndarray): Array of spatial coordinates for training data. Shape: (m_training, 2).
        training_targets (np.ndarray): Array of target values (e.g., gene expression) for training. Shape: (m_training, d).
        kernel_obj (callable): A function that computes the kernel matrix.
        effective_Laplacian (np.ndarray): The effective Laplacian matrix for the spatial graph. Shape: (m_training, m_training).
        lam_krr (float): Regularization strength for the kernel ridge regularizer.
        omega_lap (float): Regularization strength for the graph Laplacian regularizer.
        kernel_mat (np.ndarray, optional): Pre-computed kernel matrix for efficiency. Shape: (m_training, m_training).
            If None, the function will compute it.

    Returns:
        dict: A dictionary containing:
            - "theta_hat" (np.ndarray): The optimal weights vector. Shape: (m_training, d).
            - "F_hat_unnormalized_fun" (callable): A function to predict on new data.
    """
    m_training = training_spatial_locations.shape[0] # num_samples
    if kernel_mat is None:
        # Calculate the kernel matrix if not provided (note the correct normalization)
        kernel_mat = kernel_obj(training_spatial_locations) / m_training

    # Construct the system matrix for the linear solver
    system_matrix = lam_krr * np.eye(m_training) + kernel_mat + omega_lap * (effective_Laplacian @ kernel_mat)

    # Solve for the weights (theta_hat) using numpy's linear solver
    theta_hat = np.linalg.solve(kernel_mat @ system_matrix, kernel_mat @ training_targets) / np.sqrt(m_training)

    # Solution function
    def F_hat_unnormalized_fun(query_spatial_locations):
        return (kernel_obj(query_spatial_locations, training_spatial_locations) @ theta_hat) / np.sqrt(m_training)
    
    return {"theta_hat": theta_hat,
            "F_hat_unnormalized_fun": F_hat_unnormalized_fun}


def set_good_parameters_using_eigs_and_reads(training_spatial_locations, training_targets, kernel_obj, effective_Laplacian, 
                                             training_reads, omega_rel_default=6.0, fit_est_factor=0.7, 
                                             bisec_lb=1e-5, bisec_ub=1e+2, relative_buffer=1e-2, dist_reltol=1e-2):
    """
    Finds appropriate regularization strength for KRR using a bisection search.

    This function determines appropriate regularization parameters (lam_krr and omega_lap)
    by performing a bisection search to match a desired fit estimate, which is
    derived from the expected noise level based on sequencing reads.

    Args:
        training_spatial_locations (np.ndarray): Spatial coordinates for training data. Shape: (m_training, 2).
        training_targets (np.ndarray): Target values for training. Shape: (m_training, d).
        kernel_obj (callable): A function to compute the kernel matrix.
        effective_Laplacian (np.ndarray): The effective Laplacian matrix. Shape: (m_training, m_training).
        training_reads (np.ndarray): Array of read counts for each spot/pixel, used for noise estimation. Shape: (m_training,).
        omega_rel_default (float, optional): Controls the relative weight of the Laplacian regularizer.
        fit_est_factor (float, optional): Intended ratio of the data fit to the expected noise level.
        bisec_lb (float, optional): Lower bound for the bisection search.
        bisec_ub (float, optional): Upper bound for the bisection search.
        relative_buffer (float, optional): Tolerance for the bisection search.
        dist_reltol (float, optional): Relative tolerance for checking bisection convergence.

    Returns:
        dict: A dictionary with the determined parameters and final fit estimate:
            - "lam_krr" (float): The final KRR regularization parameter.
            - "omega_lap" (float): The final Laplacian regularization parameter.
            - "overall_strength_factor" (float): The overall scaling factor found by the bisection search.
            - "fit_estimate" (float): The final mean squared error.
    """
    m_training = training_spatial_locations.shape[0]
    kernel_mat = kernel_obj(training_spatial_locations) / m_training

    # Get the largest eigenvalue of the kernel matrix to set a default scaling factor
    my_res = scipy.sparse.linalg.eigsh(kernel_mat, k=1, which='LM', return_eigenvectors=False)
    lam_krr_rel_default = my_res[0]

    # Estimate the noise level from the training reads
    noise_estimate = np.mean(np.minimum(1 / training_reads, 1))
    intended_fit_estimate = fit_est_factor * noise_estimate

    # set up the bisection
    print("Bisection search for overall regularization strength ---------------------------")

    success_flag = False
    while not success_flag:
        overall_strength_iterate = (bisec_lb + bisec_ub) / 2
        sc1 = (lam_krr_rel_default) * overall_strength_iterate
        sc2 = (omega_rel_default) * overall_strength_iterate

        # Check for convergence based on relative distance
        if ((bisec_ub - bisec_lb) / overall_strength_iterate < dist_reltol):
            print('Converged but success criterion not necessarily achieved!')
            break

        # Run the KRR model with the current parameters
        spatial_results = krr_laplacian_oneshot(training_spatial_locations, training_targets, kernel_obj, effective_Laplacian, lam_krr=sc1, omega_lap=sc2, kernel_mat=kernel_mat)
        
        # Normalize the predictions and calculate the fit estimate (MSE)
        # Note: This line assumes a `utilities` module exists.
        estimate_spatial = utilities.normalize_matrix(np.maximum(spatial_results["F_hat_unnormalized_fun"](training_spatial_locations), 0), axis=1) # NOT transposed !!
        fit_estimate = (1 / m_training) * np.linalg.norm(training_targets - estimate_spatial)**2

        # print(f"Ratio = {(fit_estimate / intended_fit_estimate):.3f}")

        # Adjust the bisection bounds based on the fit estimate
        if fit_estimate > (1 + relative_buffer) * intended_fit_estimate: # regularization too strong
             bisec_ub = overall_strength_iterate
        elif fit_estimate < (1 - relative_buffer) * intended_fit_estimate: # regularization too weak
             bisec_lb = overall_strength_iterate
        else:
             success_flag = True
             print('Converged and success criterion achieved!')

    print("-------------------------------------------------------------------------------")

    return {"lam_krr": sc1, "omega_lap": sc2, "overall_strength_factor": overall_strength_iterate, "fit_estimate": fit_estimate}


# Note: Assuming 'xlogy' is imported from scipy.special or similar
# e.g., from scipy.special import xlogy

def compute_block_convex_objective_val(theta_mat, kernel_mat, W_markov, effective_Laplacian, training_spatial_locations, training_targets, 
                                       lam_krr, omega_lap, gph_s2, gph_s1):
    """
    Computes the value of the block-convex objective function for STARK.

    The objective function is composed of four terms: data fit, RKHS (ridge) regularization,
    Laplacian regularization, and graph entropy.

    Args:
        theta_mat (np.ndarray): The current KRR coefficients matrix. Shape: (m_training, 2).
        kernel_mat (np.ndarray): The kernel matrix. Shape: (m_training, m_training).
        W_markov (np.ndarray): The graph weights matrix. Shape: (m_training, m_training).
        effective_Laplacian (np.ndarray): The current effective Laplacian matrix derived from W_markov. Shape: (m_training, m_training).
        training_spatial_locations (np.ndarray): Array of spatial coordinates for training data. Shape: (m_training, 2).
        training_targets (np.ndarray): Array of target values (e.g., gene expression) for training. Shape: (m_training, d).
        lam_krr (float): Regularization strength for the kernel ridge regularizer.
        omega_lap (float): Regularization strength for the graph Laplacian regularizer.
        gph_s2 (float): The parameter s_2 in the graph weights.
        gph_s1 (float): The parameter s_1 in the graph weights.
        

    Returns:
        dict: A dictionary containing the full objective value and the value of
              each of its four component terms.
    """
    
    # Compute essential quantities
    m_training = training_spatial_locations.shape[0]
    # Reconstruct the denoised function output F (gene expression matrix)
    F_mat = np.sqrt(m_training) * (kernel_mat @ theta_mat)
    

    # 1. Data Fit Term (Mean Squared Error)
    # Measures how well the predicted values (F_mat) match the noisy targets.
    data_fit_term = (1/m_training) * np.linalg.norm(training_targets - F_mat)**2

    # 2. RKHS Regularization Term (Standard KRR penalty)
    # Penalizes the complexity of the function F in the Reproducing Kernel Hilbert Space (RKHS).
    rkhs_reg_term = lam_krr * np.sum(theta_mat * (kernel_mat @ theta_mat))

    # 3. Laplacian Regularization Term
    # Enforces spatial smoothness on the denoised function F by penalizing differences
    # between neighboring nodes as defined by the effective Laplacian.
    lap_reg_term = (omega_lap/m_training) * np.sum(F_mat * (effective_Laplacian @ F_mat))

    # Compute the pairwise spatial proximity matrix
    phi_mat = sklearn.metrics.pairwise_distances(training_spatial_locations, metric='euclidean')**2
    
    
    # 4. Graph Entropy Term
    # This term regularizes the graph weights matrix W_markov, encouraging it
    # to be smooth and to align with the spatial proximity (phi_mat).
    # NOTE: Requires the scipy.special.xlogy function for stability.
    graph_entropy_term = (0.5 * omega_lap * gph_s1**2 / m_training) * \
                         (xlogy(W_markov, W_markov).sum() - W_markov.sum() + (W_markov * phi_mat).sum()/(gph_s2**2))

    # Total Objective Value
    objective_val = data_fit_term + rkhs_reg_term + lap_reg_term + graph_entropy_term

    return {"full_objective_val": objective_val, 
            "data_fit_term": data_fit_term,
            "rkhs_reg_term": rkhs_reg_term,
            "lap_reg_term": lap_reg_term,
            "graph_entropy_term": graph_entropy_term}


def stark_block_coordinate_descent(training_spatial_locations, training_targets, kernel_obj, lam_krr, omega_lap, 
                                   gph_rho_abs, gph_s2, gph_s1=None, gex_edge_distance_quantile=0.75, num_adaptive_iter=5,
                                    compute_objective_vals=False, store_intermediate_iterates=False):
    """
    
    The alternating updates of STARK.

    This algorithm iteratively updates the KRR weights (F-step) and the graph weights (W-step) 
    to find an optimal denoising solution.

    Args:
        training_spatial_locations (np.ndarray): Array of spatial coordinates for training data. Shape: (m_training, 2).
        training_targets (np.ndarray): Array of target values (e.g., gene expression) for training. Shape: (m_training, d).
        kernel_obj (callable): A function that computes the kernel matrix.
        lam_krr (float): Regularization strength for the kernel ridge regularizer.
        omega_lap (float): Regularization strength for the graph Laplacian regularizer.
        gph_rho_abs (float): Parameter for graph edge weighting.
        gph_s2 (float): The parameter s_2 in the graph weights.
        gph_s1 (float, optional): The parameter s_1 in the graph weights. If None,
            it is computed adaptively from the data.
        gex_edge_distance_quantile (float, optional): Quantile to use for computing
            `gph_s1` if it's not provided. Defaults to 0.75.
        num_adaptive_iter (int): The number of iterations for the block-coordinate descent.
        compute_objective_vals (bool): If True, computes and stores the objective
            function value at each iteration.
        store_intermediate_iterates (bool): If True, stores the `theta` and `W_markov`
            from each iteration.

    Returns:
        dict: A dictionary containing the final denoised function F, final parameters,
              and tracking information based on the input flags.
    """

    # Initialization
    m_training = training_spatial_locations.shape[0]
    kernel_mat = kernel_obj(training_spatial_locations) / m_training
    
    # Initialize Markov graph and Laplacian from spatial coordinates alone
    W_markov = utilities.get_purely_spatial_markov_graph(training_spatial_locations, gph_rho_abs, gph_s2)
    
    adjacency_mat = (W_markov > 0)
    
    row_mean = W_markov.mean(axis=0)
    effective_Laplacian = 0.5 * (np.eye(m_training) + m_training * np.diagflat(row_mean) - W_markov - W_markov.T)

    # Initialize lists to store intermediate results if requested
    all_theta = []
    all_W_markov = []
    all_objective_vals = []
    norm_diffs_theta = []
    norm_diffs_W_markov = []

    theta_hat_prev = None
    W_markov_prev = None

    # Main block-coordinate descent loop
    for aind in range(num_adaptive_iter):

        ####### Update F given L (or W) ######
        # This is the KRR step with the Laplacian regularizer

        system_matrix = lam_krr * np.eye(m_training) + kernel_mat + omega_lap * (effective_Laplacian @ kernel_mat)
        theta_hat = np.linalg.solve(kernel_mat @ system_matrix, kernel_mat @ training_targets) / np.sqrt(m_training)
        F_hat_mat = np.sqrt(m_training) * (kernel_mat @ theta_hat) 

        # Conditionally compute gph_s1
        if gph_s1 is None:
             gph_s1 = np.quantile(utilities.compute_distances_along_edges(F_hat_mat, adjacency_mat), gex_edge_distance_quantile)
            #  print("Computing with quantiles!")

        # Store intermediate values if requested
        if compute_objective_vals:
            objective_val_here = compute_block_convex_objective_val(theta_hat, kernel_mat, W_markov, effective_Laplacian, training_spatial_locations, training_targets, 
                                       lam_krr, omega_lap, gph_s2, gph_s1)["full_objective_val"]
            print("Objective val = ", objective_val_here)
            all_objective_vals.append(objective_val_here)
        
        if store_intermediate_iterates:
            all_theta.append(theta_hat)
            all_W_markov.append(W_markov)


        # Track convergence metrics
        if theta_hat_prev is not None:
            norm_diffs_theta.append(np.linalg.norm(theta_hat - theta_hat_prev))
        
        if W_markov_prev is not None:
             norm_diffs_W_markov.append(np.linalg.norm(W_markov - W_markov_prev))

        theta_hat_prev = theta_hat.copy()
        W_markov_prev = W_markov.copy()
        
        ###### Update W given F ######
        # This is the graph update step based on the denoised expression values
        W_markov = utilities.get_markov_graph(training_spatial_locations, F_hat_mat, gph_rho_abs, gph_s2, gph_s1)
        
        row_mean = W_markov.mean(axis=0)
        effective_Laplacian = 0.5 * (np.eye(m_training) + m_training * np.diagflat(row_mean) - W_markov - W_markov.T)

        print("Done iteration: ", aind)

    # Define the final solution function to be returned
    def F_hat_unnormalized_fun(query_spatial_locations):
                return (kernel_obj(query_spatial_locations, training_spatial_locations) @ theta_hat) / np.sqrt(m_training)
    
    print("------------------------------------------------------------")
            
    # Return the results
    return {"F_hat_unnormalized_fun": F_hat_unnormalized_fun,
            "W_markov": W_markov, 
            "theta_hat": theta_hat,
            "gph_s2_1_abs": gph_s1,
            "norm_diffs_theta": norm_diffs_theta,
            "norm_diffs_W_markov": norm_diffs_W_markov,
            "all_objective_vals": all_objective_vals,
            "all_theta": all_theta,
            "all_W_markov": all_W_markov}



def stark_denoise(adata, reads_pixelwise, num_adaptive_iter=5, intended_num_neighbours=7, matern_nu=0.5, omega_rel_default=6.0, fit_est_factor=0.7, gex_edge_distance_quantile=0.75,
                  hyperparam_bisec_lb=1e-5, hyperparam_bisec_ub=20, save_to_anndata_layer=True, convergence_plots=False):
    """
    High-level wrapper for the STARK denoising pipeline, operating directly on AnnData objects.

    This function performs spatial kernel setup, optimal regularization parameter tuning,
    and executes the adaptive block-coordinate descent algorithm to denoise gene expression.

    Args:
        adata (anndata.AnnData): AnnData object containing:
            - Spatial coordinates in `adata.obsm['spatial']`.
            - Target gene expression matrix in `adata.X`.
        reads_pixelwise (np.ndarray): Total read counts per spatial spot (m_training,). Used
            to estimate the appropriate noise level for regularization tuning.
        num_adaptive_iter (int, optional): Number of iterations for the adaptive block
            coordinate descent (alternating F-step and W-step). Defaults to 5.
        intended_num_neighbours (int, optional): Target average number of spatial neighbors
            used to determine the initial spatial length scale (`length_scale`). Defaults to 7.
        matern_nu (float, optional): Parameter 'nu' for the Matern kernel, controlling
            smoothness. Common values are 0.5 (Exponential) or 1.5/2.5. Defaults to 0.5.
        omega_rel_default (float, optional): Controls the relative weight of the Laplacian regularizer.
            Defaults to 6.0
        fit_est_factor (float, optional): Intended ratio of the data fit to the expected noise level.
            Defaults to 0.7.
        gex_edge_distance_quantile (float, optional): Quantile (0 to 1) of edge distances
            in expression space used to set the initial adaptive scale parameter (gph_s1).
            Defaults to 0.75.
        hyperparam_bisec_lb (float, optional): Lower bound for the bisection search of the
            overall regularization strength. Defaults to 1e-5.
        hyperparam_bisec_ub (float, optional): Upper bound for the bisection search of the
            overall regularization strength. Defaults to 20.
        save_to_anndata_layer (bool, optional): If True, the denoised matrix is saved to
            `adata.layers['STARK_ReX']`. Defaults to True.
        convergence_plots (bool, optional): If True, displays convergence plots for the
            KRR coefficients (theta) and the graph weights (W_markov) after descent. Defaults to False.

    Returns:
        dict: A dictionary containing:
            - "F_hat_unnormalized_fun": A function that takes query spatial locations and returns
              the unnormalized denoised expression.
            - "denoised_matrix": The final denoised and normalized gene expression matrix (m_training, d).
    """
    
    # Extract spatial coordinates and data dimensions
    training_spatial_locations = np.array(adata.obsm['spatial'])
    m_training = training_spatial_locations.shape[0]
     
    print("Computing spatial length scale.") 
    # Use bisection search utility to find the radius corresponding to the intended number of neighbours
    length_scale = utilities.get_average_neighbour_radius(intended_num_neighbours, training_spatial_locations, relative_buffer=0.1, dist_reltol=1e-5)

    # Initialize the spatial kernel (Matern is used for flexibility)
    kernel_obj = kernels.Matern(length_scale=length_scale, nu=matern_nu)

    # Define the initial spatial scale parameter (gph_s2) for the graph kernel
    gph_s2 = length_scale
    gph_s1 = None # gph_s1 is initialized to None and will be tuned adaptively
    
    # Define the hard cutoff for spatial adjacency
    graph_threshold_rad_ratio = 1.5
    gph_rho_abs = length_scale * graph_threshold_rad_ratio

    # Initialize the purely spatial Markov graph and its effective Laplacian
    W_markov = utilities.get_purely_spatial_markov_graph(training_spatial_locations, gph_rho_abs, gph_s2)
    row_mean = W_markov.mean(axis=0)
    effective_Laplacian_spatial = 0.5 * (np.eye(m_training) + m_training * np.diagflat(row_mean) - W_markov - W_markov.T)

    # Extract target gene expression data
    training_targets = adata.X

    print("Computing suitable hyperparameters.")
    # Use bisection search to find the overall regularization strength (lam_krr and omega_lap)
    # that matches the target fit estimate derived from noise (reads_pixelwise)
    param_results = set_good_parameters_using_eigs_and_reads(training_spatial_locations, training_targets, kernel_obj, effective_Laplacian_spatial, reads_pixelwise,
                                                                            omega_rel_default=omega_rel_default, fit_est_factor=fit_est_factor, relative_buffer=0.01, 
                                                                            bisec_lb=hyperparam_bisec_lb, bisec_ub=hyperparam_bisec_ub)
    lam_krr = param_results['lam_krr']
    omega_lap = param_results['omega_lap']


    print("Running block coordinate descent.")

    # Execute the core adaptive optimization algorithm
    algo_results = stark_block_coordinate_descent(training_spatial_locations, training_targets, kernel_obj, lam_krr, omega_lap, 
                                   gph_rho_abs, gph_s2, gph_s1=gph_s1, gex_edge_distance_quantile=gex_edge_distance_quantile, 
                                   num_adaptive_iter=num_adaptive_iter, compute_objective_vals=False, store_intermediate_iterates=False)
    
    # Compute final estimate and normalize to simpleces
    denoised_matrix = utilities.normalize_matrix(np.maximum(algo_results["F_hat_unnormalized_fun"](training_spatial_locations), 0), axis=1) # NOT transposed !!

    # Save results back to the AnnData object layer
    if save_to_anndata_layer:
        adata.layers["STARK_ReX"] = denoised_matrix.copy()

    # Generate and display convergence plots if requested
    if convergence_plots:
        fig_conv = plt.figure(figsize=(10,4))
        
        # Plot 1: Convergence of KRR coefficients (theta)
        plt.subplot(1,2,1)
        plt.stem(algo_results['norm_diffs_theta'])
        plt.yscale('log')
        plt.xlabel("Iterations")
        plt.ylabel(r"$\|\theta^{t+1} - \theta^t\|_F$")

        # Plot 2: Convergence of the graph weights
        plt.subplot(1,2,2)
        plt.stem(algo_results['norm_diffs_W_markov'])
        plt.yscale('log')
        plt.xlabel("Iterations")
        plt.ylabel(r"$\|W^{t+1} - W^t\|_F$")

        fig_conv.tight_layout()
        plt.show()

    # Return the necessary results
    return {"F_hat_unnormalized_fun": algo_results["F_hat_unnormalized_fun"],
            "denoised_matrix": denoised_matrix}


    
