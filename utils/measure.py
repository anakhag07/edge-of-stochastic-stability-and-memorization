import torch as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import linalg as LA
import numpy as np
from typing import List, Optional, Tuple

import wandb
from .lobpcg import torch_lobpcg, _maybe_orthonormalize
# from .hvp import make_param_block_hvp
from torch.func import functional_call

import time
import os
from scipy import stats

__all__ = ['compute_train_test_gap_from_tensors','param_vector', 'param_length', 'flatt', 'grads_vector', 
           'calculate_all_the_grads', 'compute_eigenvalues', 'compute_grad_H_grad', 
           'calculate_averaged_lambdamax', 'create_ntk', 
           'compute_fisher_eigenvalues', 'calculate_all_net_grads',
           'calculate_averaged_grad_H_grad', 'calculate_averaged_grad_H_grad_step', 'calculate_gni',
           'calculate_accuracy', 'calculate_param_distance',
           'EigenvectorCache', 'create_hessian_vector_product', 'compute_multiple_eigenvalues_lobpcg',
           'calculate_gradient_norm_squared_mc', 'calculate_expected_one_step_full_loss_change',
           'calculate_expected_one_step_batch_loss_change', 'compute_gradient_projection_ratios',
           'estimate_hessian_trace', 'gimme_new_rng', 'gimme_random_subset_idx',
           'compute_per_example_losses', 'compute_outlier_vs_bulk_stats_hessian',
           'extract_feature_matrix', 'identify_knn_outliers_by_neighbor_mix',
           'select_dataset_subset', 'compute_subset_metrics', 'compute_subset_metrics_from_tensors']


class EigenvectorCache:
    """
    A cache for storing eigenvectors to enable warm starts in power iteration methods.
    Designed to be compatible with future LOBPCG implementations.
    """
    def __init__(self, max_eigenvectors=5):
        self.max_eigenvectors = max_eigenvectors
        self.eigenvectors = []   # List of eigenvectors for multi-eigenvalue computations
        self.eigenvalues = []    # Corresponding eigenvalues
        
    def store_eigenvector(self, eigenvector, eigenvalue=None):
        """Store a single eigenvector (and optionally eigenvalue)"""
        if eigenvalue is not None:
            self.eigenvalues = [eigenvalue]
        self.eigenvectors = [eigenvector]
    
    def store_eigenvectors(self, eigenvectors_list, eigenvalues_list=None):
        """Store multiple eigenvectors (for future LOBPCG compatibility)"""
        self.eigenvectors = [v.detach().clone() for v in eigenvectors_list]
        if eigenvalues_list is not None:
            self.eigenvalues = list(eigenvalues_list)
        
        # Trim to maximum size
        if len(self.eigenvectors) > self.max_eigenvectors:
            self.eigenvectors = self.eigenvectors[:self.max_eigenvectors]
            if self.eigenvalues:
                self.eigenvalues = self.eigenvalues[:self.max_eigenvectors]
    
    def get_warm_start_vectors(self, device=None):
        """Get eigenvectors for warm start, optionally moved to specified device"""
        if not self.eigenvectors:
            return None
        
        if device is not None:
            return [v.to(device) for v in self.eigenvectors]
        return self.eigenvectors
    
    def clear(self):
        """Clear all cached eigenvectors"""
        self.eigenvectors = []
        self.eigenvalues = []
    
    def __len__(self):
        return len(self.eigenvectors)
    
    def __contains__(self, key):
        # For backward compatibility with dict-like access
        return hasattr(self, key) and getattr(self, key) is not None



################################################################################
#                                                                              #
#                               HELPER FUNCTIONS                               #
#                                                                              #
################################################################################


def param_vector(net, clone=True):
    '''
    Returns a vector of all the parameters of the network
    If clone=True, returns a detached clone of the parameters
    '''
    # params = list(net.parameters())
    param_vector = T.cat([p.flatten() for p in net.parameters()])
    if clone:
        return param_vector.detach().clone()
    return param_vector

def param_length(net):
    '''
    Returns the number of parameters in the network
    '''
    params = list(net.parameters())
    return sum([p.numel() for p in params])

def flatt(vectors):
    '''
    Flattens a list of vectors into a single vector
    '''
    return T.cat([v.flatten() for v in vectors])


def grads_vector(net):  
    # pull out all the gradients from a network as one vector
    grads = []
    for p in net.parameters():
        grads.append(p.grad.flatten().detach().clone())
    return T.cat(grads)


def gimme_new_rng():
    """
    Create a new random number generator with a unique seed.
    """
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    return rng


def gimme_random_subset_idx(dataset_size, subset_size):
    """
    Get random indices for a subset of the dataset.

    Args:
        dataset_size (int): Total size of the dataset.
        subset_size (int): Desired size of the subset.

    Returns:
        Tensor: Random indices for the subset.
    """
    rng = gimme_new_rng()

    shuffle = T.randperm(dataset_size, generator=rng)
    random_idx = shuffle[:subset_size]
    return random_idx


def calculate_param_distance(net, reference_params, p=2):
    """
    Calculate the distance between current network parameters and reference parameters.
    
    Args:
        net (nn.Module): Neural network model
        reference_params (Tensor): Flattened reference parameters (from param_vector())
        p (int, optional): The norm degree. Default: 2 for Euclidean distance
    
    Returns:
        Tensor: The p-norm distance between current and reference parameters
    """
    with torch.no_grad():
        current_params = param_vector(net)
        return T.linalg.vector_norm(current_params - reference_params, ord=p)


def calculate_all_the_grads(net, X, Y, loss_fn, optimizer, storage_device=None):
    # device = net.parameters().__next__().device

    grads = [] # datapoint, parameter
    for x, y in zip(X, Y):
        optimizer.zero_grad()
        y_pred = net(x.unsqueeze(0)).squeeze(dim=-1)
        loss = loss_fn(y_pred, y.unsqueeze(0))
        loss.backward()
        detached_grads = grads_vector(net).detach()
        if storage_device:
            detached_grads = detached_grads.to(storage_device)
        grads.append(detached_grads)
    
    return T.stack(grads)


def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy given the model predictions and target labels.
    
    Args:
        predictions: tensor of shape (num_samples, num_classes) with model predictions
        targets: tensor of shape (num_samples, num_classes) with one-hot encoded labels
                or tensor of shape (num_samples,) with class indices
    
    Returns:
        accuracy: float representing the accuracy (0.0 to 1.0)
    """
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Get the predicted class (highest value in each row)
        # this is if we have all the classes
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        # Get the predicted class (sign of the prediction)
        # this is if we have only two classes
        pred_classes = torch.sign(predictions).long()
    
    # Check if targets are one-hot encoded or class indices
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        # One-hot encoded targets
        true_classes = torch.argmax(targets, dim=1)
    else:
        # Class indices (1D tensor)
        if len(targets.shape) == 1:
            true_classes = torch.round(targets).long()
        else:
            true_classes = targets.long()
    
    # Compare and compute accuracy
    correct = (pred_classes == true_classes).sum().item()
    total = targets.size(0)
    
    return correct / total


def jvp(net, X, Y, loss_fn, vector):
    """
    Computes the Jacobian-vector product (JVP) of the loss with respect to the network parameters.
    
    Args:
        net (nn.Module): The neural network model
        X (Tensor): Input data
        Y (Tensor): Target labels
        loss_fn (callable): Loss function to compute the loss
    
    Returns:
        Tensor: The JVP of the loss with respect to the network parameters
    """
    params = list(net.parameters())
    y_pred = net(X).squeeze(dim=-1)
    loss = loss_fn(y_pred, Y, sampling_vector=vector)
    
    # Compute gradients
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    # Flatten gradients into a single vector
    grads_vector = flatt(grads).detach()
    return grads_vector



################################################################################
#                                                                              #
#                             EIGENVALUE FUNCTIONS                             #
#                                                                              #
################################################################################


def compute_eigenvalues(loss, 
                        net, 
                        k=1, 
                        max_iterations=100, 
                        reltol=1e-2,
                        init_vectors=None,
                        batched=None,
                        eigenvector_cache=None,
                        return_eigenvectors: bool = False,
                        use_power_iteration: bool = False):
    """
    Computes the top-k eigenvalues of the Hessian of the loss function at the current point.
    
    Uses LOBPCG by default for better performance, with power iteration as fallback for k=1.

    Args:
        loss (Tensor): The loss value at the current point
        net (nn.Module): The neural network model
        k (int, optional): Number of eigenvalues to compute. Defaults to 1.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 1000.
        reltol (float, optional): relative tolerance threshold for eigenvalue computation. Defaults to 1e-2.
        init_vectors (Tensor, optional): Initial vectors. For k=1, can be 1D vector. For k>1, should be [n_params, k]. 
                                        If None, uses cached or random vectors. Defaults to None.
        batched (Any, optional): Unused parameter. Defaults to None.
        eigenvector_cache (EigenvectorCache, optional): Cache to store/retrieve eigenvectors for warm starts. Defaults to None.
        return_eigenvectors (bool, optional): Whether to return the final eigenvectors. Defaults to False.
        use_power_iteration (bool, optional): If True, force use of power iteration (only works for k=1). Defaults to False.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If k=1 and return_eigenvectors=False: Returns single eigenvalue (scalar Tensor)
            - If k=1 and return_eigenvectors=True: Returns (eigenvalue, eigenvector)
            - If k>1 and return_eigenvectors=False: Returns eigenvalues tensor of shape [k]
            - If k>1 and return_eigenvectors=True: Returns (eigenvalues, eigenvectors) where 
              eigenvalues has shape [k] and eigenvectors has shape [n_params, k]

    Note:
        By default, uses LOBPCG for eigenvalue computation for better performance.
        Falls back to power iteration if use_power_iteration=True (only supported for k=1).
        
        If eigenvector_cache is provided, the function will try to reuse previous eigenvectors
        for warm starts and store the final eigenvector(s) for future use.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    
    if use_power_iteration and k > 1:
        raise ValueError("Power iteration only supports k=1. Use LOBPCG (default) for k>1.")
    
    device = next(net.parameters()).device

    # Choose method: use LOBPCG by default unless explicitly requested to use power iteration
    if use_power_iteration and k == 1:
        # Use the existing power iteration implementation
        return compute_lambdamax_power_iteration(
            loss, net, max_iterations, reltol, init_vectors, batched,
            eigenvector_cache, return_eigenvectors
        )
    else:
        # Use LOBPCG method (default)
        eigenvalues, eigenvectors = compute_multiple_eigenvalues_lobpcg(
            loss, net, k, max_iterations, reltol, init_vectors, 
            eigenvector_cache, return_eigenvectors=True
        )
        
        if k == 1:
            # For backward compatibility with single eigenvalue case
            eigenvalue = eigenvalues[0]
            if return_eigenvectors:
                return eigenvalue, eigenvectors[:, 0]
            else:
                return eigenvalue
        else:
            # Multiple eigenvalues case
            if return_eigenvectors:
                return eigenvalues, eigenvectors
            else:
                return eigenvalues


def create_hessian_vector_product(loss, net):
    """
    Create a Hessian-vector product function for use with LOBPCG.
    
    This function creates a closure that computes the Hessian-vector product
    H @ v where H is the Hessian of the loss function with respect to network parameters.
    
    Args:
        loss (Tensor): The loss value at the current point (must retain computational graph)
        net (nn.Module): The neural network model
        
    Returns:
        callable: A function that takes a vector v and returns H @ v
    """
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_vector = flatt(grads)
    
    def hessian_vector_product(v):
        """
        Compute Hessian-vector product H @ v.
        
        Args:
            v (Tensor): Vector(s) to multiply with Hessian. Can be 1D or 2D (for multiple vectors).
            
        Returns:
            Tensor: H @ v (same shape as v)
        """
        # Handle both 1D and 2D inputs for compatibility with LOBPCG
        if v.dim() == 1:
            # Single vector case
            grad_v = torch.dot(grads_vector, v)
            Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
            return flatt(Hv)
        elif v.dim() == 2:
            # Multiple vectors case (for LOBPCG)
            results = []
            for i in range(v.shape[1]):
                vi = v[:, i]
                grad_v = torch.dot(grads_vector, vi)
                Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
                results.append(flatt(Hv))
            return torch.stack(results, dim=1)
        else:
            raise ValueError(f"Input tensor must be 1D or 2D, got {v.dim()}D")
    
    return hessian_vector_product


def compute_multiple_eigenvalues_lobpcg(loss, net, k=5, max_iterations=100, reltol=1e-2,
                                       init_vectors=None, eigenvector_cache=None,
                                       return_eigenvectors=False):
    """
    Compute multiple eigenvalues of the Hessian using LOBPCG algorithm.
    
    This function computes the top-k eigenvalues of the Hessian matrix using the
    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) algorithm.
    
    Args:
        loss (Tensor): The loss value at the current point (must retain computational graph)
        net (nn.Module): The neural network model
        k (int, optional): Number of eigenvalues to compute. Defaults to 5.
        max_iterations (int, optional): Maximum number of LOBPCG iterations. Defaults to 100.
        reltol (float, optional): Relative tolerance for LOBPCG convergence. Defaults to 2% relative tolerance.

        init_vectors (Tensor, optional): Initial vectors for LOBPCG (shape: [n_params, k]). 
                                       If None, uses random or cached vectors.
        eigenvector_cache (EigenvectorCache, optional): Cache for storing/retrieving eigenvectors.
        return_eigenvectors (bool, optional): Whether to return eigenvectors along with eigenvalues.
        
    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If return_eigenvectors is False: Returns eigenvalues tensor of shape [k]
            - If return_eigenvectors is True: Returns tuple of (eigenvalues, eigenvectors)
              where eigenvectors has shape [n_params, k]
    
    Note:
        The eigenvalues are returned in descending order (largest first).
        The function automatically handles the case where k is too large relative to the problem size.
    """
    device = next(net.parameters()).device
    n_params = param_length(net)
    
    # Create Hessian-vector product function
    hessian_matvec = create_hessian_vector_product(loss, net)
    
    # Initialize vectors with priority: init_vectors > cached vectors > random
    if init_vectors is not None:
        X = init_vectors
        if X.shape[1] != k:
            raise ValueError(f"init_vectors must have shape [n_params, {k}], got {X.shape}")
    elif eigenvector_cache is not None and len(eigenvector_cache) > 0:
        # Use cached eigenvectors as initial guess
        cached_vectors = eigenvector_cache.get_warm_start_vectors(device)
        if cached_vectors:
            # Take up to k vectors from cache, pad with random if needed
            n_cached = min(len(cached_vectors), k)
            X_list = cached_vectors[:n_cached]
            
            # Pad with random vectors if we don't have enough cached vectors
            if n_cached < k:
                n_random = k - n_cached
                random_vectors = torch.randn(n_params, n_random, device=device)
                X_list.extend([random_vectors[:, i] for i in range(n_random)])
            
            X = torch.stack(X_list, dim=1)
        else:
            X = torch.randn(n_params, k, device=device)
    else:
        # Use random initialization
        X = torch.randn(n_params, k, device=device)
    
    # Ensure X is on the correct device and has the right shape
    X = X.to(device)
    if X.shape != (n_params, k):
        X = X.reshape(n_params, k)
    
    # Run LOBPCG
    tol = reltol / (20 * n_params)  # Adjust tolerance based on problem size

    eigenvalues, eigenvectors, iterations = torch_lobpcg(
        hessian_matvec, X, max_iter=max_iterations, tol=tol
    )
    
    # Log the number of iterations to wandb (if available)
    try:
        wandb.log({"lobpcg_iterations": iterations}, commit=False)
    except:
        pass  # wandb not initialized or not available
    
    # Store eigenvectors in cache for future use
    if eigenvector_cache is not None:
        eigenvector_list = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]
        eigenvector_cache.store_eigenvectors(eigenvector_list, eigenvalues.tolist())
    
    # Return results
    if return_eigenvectors:
        return eigenvalues, eigenvectors
    else:
        return eigenvalues



def compute_lambdamax_power_iteration(loss, net, max_iterations, reltol, init_vector,
                                       eigenvector_cache, return_eigenvector):
    """Power iteration implementation of the maximum eigenvalue of the Hessian."""
    device = next(net.parameters()).device

    # compute gradient and keep it
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_vector = flatt(grads)

    size = param_length(net)
    
    # Initialize vector with priority: init_vector > cached eigenvector > gradient
    if init_vector is not None:
        v = init_vector
    elif eigenvector_cache is not None:
        # Support both EigenvectorCache objects and dict-style caches
        if isinstance(eigenvector_cache, EigenvectorCache):
            if len(eigenvector_cache) > 0:
                cached_v = eigenvector_cache.eigenvector
                if cached_v.device != device:
                    cached_v = cached_v.to(device)
                v = cached_v.detach()
            else:
                v = T.randn(size, device=device)
        elif isinstance(eigenvector_cache, dict) and 'eigenvector' in eigenvector_cache:
            # Backward compatibility with dict-style cache
            cached_v = eigenvector_cache['eigenvector']
            if cached_v.device != device:
                cached_v = cached_v.to(device)
            v = cached_v.detach()
        else:
            v = T.randn(size, device=device)
    else:
        # Use random vector as initial vector instead of gradient
        v = T.randn(size, device=device)
    
    with torch.no_grad():
        v = v / T.linalg.norm(v)



    # grad_vector as init_vector, since it is very close to the eigenvector
    # v = grads_vector.detach()

    # Power iteration method to find the maximum eigenvalue
    # NEW METHOD
    v = v.detach()
    eigenval = 0.0  # Initialize eigenval to avoid undefined variable error
    for i in range(max_iterations):
        # grad_vector = \nabla L
        grad_v = T.dot(grads_vector, v) # \nabla L . v
        Hv = flatt(T.autograd.grad(grad_v, params, retain_graph=True)).detach() # \nabla (\nabla L . v) = H(L) * v

        v = v.detach()
        with T.no_grad():
            rayleigh_quotient = T.dot(Hv, v) / T.dot(v, v)
            eigenval = rayleigh_quotient  # Update eigenval every iteration
            if T.abs(rayleigh_quotient) < 1e-12:
                break

            residual = Hv - rayleigh_quotient * v
            resid_norm = T.linalg.norm(residual)
            if resid_norm / T.abs(rayleigh_quotient) < reltol:
                break
            
            v = Hv / T.linalg.norm(Hv) # Normalize for next iteration, 
    
    
        
    #### OLD PROCEDURE
    #### This is the old procedure, kept for reference
    # epsilon = 1e-4
    # eigenval = -10
    # eigenvals = [-100, -20]
    # for i in range(max_iterations):
    #     grad_v = T.dot(grads_vector, v)
    #     Hv = flatt(T.autograd.grad(grad_v, params, retain_graph=True))
    #     with T.no_grad():
    #         v = Hv / T.linalg.vector_norm(Hv)
    #         v = v.detach()
    #         eigenval = T.dot(Hv, v) / T.dot(v, v)
    #     old_eigenval = eigenvals[-2]
    #     eigenvals.append(eigenval)

    #     if abs(eigenval - old_eigenval) / eigenval < epsilon:
    #         break

    # Log the number of iterations to wandb
    try:
        wandb.log({"power_iteration_iterations": i + 1}, commit=False)
    except:
        pass



    # Store the final eigenvector in cache for future warm starts
    if eigenvector_cache is not None:
        if isinstance(eigenvector_cache, EigenvectorCache):
            eigenvector_cache.store_eigenvector(v, eigenval)
        else:
            raise ValueError("eigenvector_cache must be an instance of EigenvectorCache")

    # Prepare return values
    results = [eigenval]
    
    if return_eigenvector:
        results.append(v.detach())
    
    # Return single value or tuple based on what was requested
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


################################################################################
#                                                                              #
#                         GRAD-H-GRAD (BATCH SHARPNESS)                        #
#                                                                              #
################################################################################


def compute_grad_H_grad(loss, net, grad_already_there: bool = False,
                        return_ghg_gg_separately: bool = False):
    """
    Computes g^T H g / ||g||², the Rayleigh quotient of the Hessian H and gradient g.
    
    This function calculates gradient * Hessian * gradient normalized by the squared gradient norm,
    which represents the curvature of the loss in the gradient direction. If taken on a batch, this is 
    step sharpness. Averaging over many batches gives batch sharpness.
    
    Args:
        loss (Tensor): Loss value (must retain computational graph for Hessian computation)
        net (nn.Module): Neural network model
        grad_already_there (bool, optional): Use existing gradients instead of computing new ones. Defaults to False.
        return_ghg_gg_separately (bool, optional): Return (g^T H g, g^T g) separately instead of ratio. Defaults to False.
    
    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]: Rayleigh quotient g^T H g / ||g||² or separate components if requested
    """
    
    device = next(net.parameters()).device

    # compute gradient and keep it
    params = list(net.parameters())
    if not grad_already_there:
        grads = torch.autograd.grad(loss, params, create_graph=True)
    else:
        grads = [p.grad for p in params]
    grads_vector = flatt(grads)

    # compute Hessian vector product
    # grads_vector = T.cat([g.flatten() for g in grads])
    step_vector = grads_vector.detach()
    grad_step = T.dot(grads_vector, step_vector)
    Hv = T.autograd.grad(grad_step, params, retain_graph=False)
    Hv = flatt(Hv).detach()

    if return_ghg_gg_separately:
        return T.dot(step_vector, Hv), T.dot(step_vector, step_vector)
    return T.dot(step_vector, Hv) / T.dot(step_vector, step_vector)



def calculate_averaged_grad_H_grad(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              eps = 0.005, # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96,
                              expectation_inside = False,
                              with_replacement = False,
                              return_confidence_interval: bool = False,
                              confidence_level: float = 0.95
                              ): 
    """
    Computes E[g_b H_b g_b / ||g_b||²], which represents batch sharpness, aka the Rayleigh quotient of the 
    batch Hessian and batch gradient.
    The function uses Monte Carlo sampling with adaptive stopping based on relative standard 
    error to efficiently estimate the expectation.
    Args:
        net: Neural network model whose parameters will be used for gradient computation
        X: Input data tensor
        Y: Target labels tensor  
        loss_fn: Loss function to compute gradients from
        batch_size (int): Size of random batches to sample for each estimate
        n_estimates (int, optional): Maximum number of Monte Carlo estimates. Defaults to 500.
        min_estimates (int, optional): Minimum estimates before checking stopping criterion. Defaults to 10.
        eps (float, optional): Relative standard error threshold for early stopping. Defaults to 0.005.
        expectation_inside (bool, optional): If True, computes E[gHg]/E[g²] instead, mostly used for exploratory purposes. Defaults to False.
        with_replacement (bool, optional): Sample batches with replacement. Defaults to False.
        return_confidence_interval (bool, optional): If True, include a confidence interval and related statistics in the return value. Defaults to False.
        confidence_level (float, optional): Confidence level for the interval when `return_confidence_interval` is True. Defaults to 0.95.
    Returns:
        float or dict: The averaged gradient-Hessian-gradient ratio representing batch sharpness. When
            `return_confidence_interval` is True, returns a dictionary with the estimate, confidence interval,
            standard error, confidence level, and number of Monte Carlo samples used.
    Notes:
        - Uses independent random number generator for true randomness (since it is fixed in the main training loop)
        - Implements adaptive stopping based on relative standard error convergence  
        - Logs the number of estimates to wandb if available
        - eps=0.005 approximately gives 1% estimation error
    """
    gHg_vals = []
    norm_g_vals = []

    x_vals = gHg_vals
    y_vals = norm_g_vals
    

    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)

    for i in range(n_estimates):
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        if with_replacement:
            random_idx = T.randint(0, len(X), (batch_size,), generator=rng)
            
        if batch_size > 128:
            torch.cuda.empty_cache()
         
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]

        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        
        gHg, norm_g = compute_grad_H_grad(loss, net, return_ghg_gg_separately=True)
        gHg = gHg.item()
        norm_g = norm_g.item()
        
        
        gHg_vals.append(gHg)
        norm_g_vals.append(norm_g)

        if i < min_estimates:
            continue    

        mean_x, mean_y = np.mean(x_vals), np.mean(y_vals)
        var_x,  var_y  = np.var(x_vals, ddof=1), np.var(y_vals, ddof=1)
        cov_xy = np.cov(x_vals, y_vals, ddof=1)[0, 1]

        R = mean_x / mean_y

        var_R = (var_x / mean_y**2
                 - 2 * cov_xy * mean_x / mean_y**3
                 + var_y * mean_x**2 / mean_y**4) / i

        rse = np.sqrt(var_R) / abs(R)  # relative standard error

        if rse < eps:                    # stopping rule
            break


    num_samples = len(gHg_vals)

    try:
        wandb.log({"number_of_gHg_estimates": num_samples}, commit=False)
    except:
        pass


    if num_samples == 0:
        raise RuntimeError("calculate_averaged_grad_H_grad received no samples; check dataset and parameters.")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    alpha = 1 - confidence_level

    if expectation_inside:
        mean_x = float(np.mean(gHg_vals))
        mean_y = float(np.mean(norm_g_vals))
        if mean_y == 0.0:
            raise ZeroDivisionError("Mean squared gradient is zero; cannot compute batch sharpness.")

        result = mean_x / mean_y

        if not return_confidence_interval:
            return result

        if num_samples < 2:
            stderr = 0.0
            ci = (result, result)
        else:
            var_x = float(np.var(gHg_vals, ddof=1))
            var_y = float(np.var(norm_g_vals, ddof=1))
            cov_xy = float(np.cov(gHg_vals, norm_g_vals, ddof=1)[0, 1])
            var_R = (
                var_x / (mean_y ** 2)
                - 2 * cov_xy * mean_x / (mean_y ** 3)
                + var_y * (mean_x ** 2) / (mean_y ** 4)
            ) / num_samples
            var_R = max(var_R, 0.0)
            stderr = float(np.sqrt(var_R))
            t_multiplier = stats.t.ppf(1 - alpha / 2, df=num_samples - 1) if num_samples > 1 else 0.0
            if not np.isfinite(t_multiplier):
                t_multiplier = 0.0
            half_width = float(t_multiplier * stderr)
            ci = (result - half_width, result + half_width)

        return {
            "mean": result,
            "ci": ci,
            "stderr": stderr,
            "confidence_level": confidence_level,
            "num_samples": num_samples,
        }

    gHg_normalized = np.array(gHg_vals) / np.array(norm_g_vals)
    result = float(np.mean(gHg_normalized))

    if not return_confidence_interval:
        return result

    if num_samples < 2:
        stderr = 0.0
        ci = (result, result)
    else:
        std = float(np.std(gHg_normalized, ddof=1))
        stderr = float(std / np.sqrt(num_samples))
        t_multiplier = stats.t.ppf(1 - alpha / 2, df=num_samples - 1)
        if not np.isfinite(t_multiplier):
            t_multiplier = 0.0
        half_width = float(t_multiplier * stderr)
        ci = (result - half_width, result + half_width)

    return {
        "mean": result,
        "ci": ci,
        "stderr": stderr,
        "confidence_level": confidence_level,
        "num_samples": num_samples,
    }


def calculate_averaged_grad_H_grad_step(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 1000,
                              min_estimates = 10,
                              eps = 0.005,
                              log_the_expectation_outside = False,
                              return_ghg_gg_separately = False,
                              with_replacement = False,
                              return_confidence_interval: bool = False,
                              confidence_level: float = 0.95
                              ):
    """Backward-compatible wrapper for the batch sharpness estimator E[gHg/g²]."""
    if return_ghg_gg_separately:
        raise NotImplementedError("Returning gHg and g² separately is not supported in this refactor.")

    result = calculate_averaged_grad_H_grad(
        net=net,
        X=X,
        Y=Y,
        loss_fn=loss_fn,
        batch_size=batch_size,
        n_estimates=n_estimates,
        min_estimates=min_estimates,
        eps=eps,
        expectation_inside=False,
        with_replacement=with_replacement,
        return_confidence_interval=return_confidence_interval,
        confidence_level=confidence_level,
    )

    return result


################################################################################
#                                                                              #
#                       GRADIENT–NOISE INTERACTION (GNI)                       #
#                                                                              #
################################################################################


def calculate_gni(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                            #   max_hessian_iters = 1000,
                            #   hessian_tolerance = 1e-3,
                              batched = None,
                              compute_gHg: bool = False,
                              use_subset_of_data: int = None # use only a subset of the dataset to calculate H in GNI - speeds up computations!
                              ): 
    sharpnesses = []

    params = list(net.parameters())


    if use_subset_of_data is not None:
        rng = gimme_new_rng()
        # Take random subset of the dataset
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:use_subset_of_data]
        X = X[random_idx]
        Y = Y[random_idx]

    total_loss = loss_fn(net(X).squeeze(dim=-1), Y)

    total_grad = flatt(torch.autograd.grad(total_loss, params, create_graph=True))

    total_grad_detach = total_grad.detach()

    normalizer = T.dot(total_grad_detach, total_grad_detach).item()

    gHg_list = []


    for i in range(n_estimates):
        rng = gimme_new_rng()

        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]


        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        grads_vector = flatt(torch.autograd.grad(loss, params))
        step_vector = grads_vector.detach()

        total_grad_dot_step = T.dot(total_grad, step_vector)

        Hg = flatt(torch.autograd.grad(total_grad_dot_step, params, retain_graph=True))

        Hg = Hg.detach()

        gHg = T.dot(step_vector, Hg)

        gHg_list.append(gHg.item())


    quantity = np.mean(gHg_list) / normalizer

    return quantity



################################################################################
#                                                                              #
#                               MISCELLANEOUS                                  #
#                                                                              #
################################################################################


def compute_gradient_projection_ratios(grad_vector: torch.Tensor,
                                       eigvecs: torch.Tensor,
                                       max_k: int = 20,
                                       eigenvalues: list = None) -> dict:
    """
    Compute cumulative projection ratios of the full-batch gradient onto the
    subspace spanned by the top-i eigenvectors, i = 1..k, where k = min(K, max_k).

    grad_projection_i = ||Proj_{span(v1..vi)}(g)||_2 / ||g||_2

    Args:
        grad_vector: Flattened full-batch gradient g, shape [n]
        eigvecs:   Matrix of eigenvectors, shape [n, K]
        max_k:     Cap on how many cumulative projections to report (default 20)
        eigenvalues: Optional list of eigenvalues (length K) to ensure proper
                     descending ordering; if provided, will sort eigvecs by it.

    Returns:
        dict mapping names 'grad_projection_01', ..., 'grad_projection_{k:02d}',
        and 'grad_projection_residual' to floats in [0, 1].

    Notes:
        - Uses _maybe_orthonormalize to cheaply verify and, if needed,
          re-orthonormalize the eigenvector block prior to projection.
        - If grad_vector has zero norm, returns all zeros.
    """
    if grad_vector is None or eigvecs is None:
        return {}

    # Ensure 2D [n, K]
    if eigvecs.dim() == 1:
        eigvecs = eigvecs.unsqueeze(1)

    n, K = eigvecs.shape
    if n != grad_vector.numel():
        raise ValueError(f"Dimension mismatch: gradient has {grad_vector.numel()} params, eigenvectors have {n}")

    # Limit to at most max_k eigenvectors
    k = min(K, max_k)

    # If eigenvalues are supplied, sort eigenvectors by descending eigenvalue
    if eigenvalues is not None and len(eigenvalues) >= k:
        # Sort pairs (eigenvalue, column index) descending by value
        import math
        order = sorted(range(len(eigenvalues)), key=lambda idx: (-float(eigenvalues[idx]) if not math.isnan(float(eigenvalues[idx])) else float('inf')))
        order = order[:k]
        V = eigvecs[:, order]
    else:
        V = eigvecs[:, :k]

    # Quick orthonormality check; orthonormalize if necessary
    V = _maybe_orthonormalize(V, assume_ortho=True)

    # Compute projection coefficients c = V^T s
    g = grad_vector.reshape(-1)
    g_norm = torch.linalg.vector_norm(g)
    if g_norm.item() == 0.0:
        # Degenerate step; return zeros
        result = {f"grad_projection_{i:02d}": 0.0 for i in range(1, k + 1)}
        result["grad_projection_residual"] = 0.0
        return result

    c = V.T @ g  # shape [k]
    c2 = c.pow(2)
    # Cumulative squared projection norms
    c2_cum = torch.cumsum(c2, dim=0)
    denom = g_norm.pow(2)
    # Convert to ratios in [0,1]
    ratios = torch.sqrt(torch.clamp(c2_cum / denom, min=0.0, max=1.0))

    result = {}
    for i in range(k):
        result[f"grad_projection_{i+1:02d}"] = float(ratios[i].item())

    # Residual norm ratio for the full k-dimensional subspace
    residual_sq = torch.clamp(1.0 - c2_cum[-1] / denom, min=0.0, max=1.0)
    result["grad_projection_residual"] = float(torch.sqrt(residual_sq).item())

    return result


def estimate_hessian_trace(net,
                           X,
                           Y,
                           loss_fn,
                           max_estimates: int = 512,
                           min_estimates: int = 10,
                           eps: float = 0.01,
                           generator: Optional[torch.Generator] = None,
                           probe_type: str = 'rademacher') -> float:
    """
    Estimate the trace of the full-batch loss Hessian via Hutchinson's method.

    Args:
        net: Neural network model.
        X: Full input tensor used to construct the loss.
        Y: Full target tensor used to construct the loss.
        loss_fn: Callable loss function applied on the full batch.
        max_estimates: Maximum number of probe vectors to use.
        min_estimates: Minimum number of probes before adaptive stopping is checked.
        eps: Relative standard error tolerance for adaptive stopping.
        generator: Optional RNG to make the estimator deterministic (useful in tests).
        probe_type: Distribution for probe vectors. Currently only 'rademacher' is supported.

    Returns:
        float: Estimated trace of the Hessian.
    """

    if max_estimates < 1:
        raise ValueError("max_estimates must be positive")
    if min_estimates < 1:
        raise ValueError("min_estimates must be positive")
    if min_estimates > max_estimates:
        raise ValueError("min_estimates cannot exceed max_estimates")
    if probe_type != 'rademacher':
        raise NotImplementedError(f"Unsupported probe_type: {probe_type}")

    first_param = next(net.parameters())
    device = first_param.device
    dtype = first_param.dtype

    # Evaluate full-batch loss and build Hessian-vector product closure
    preds = net(X).squeeze(dim=-1)
    loss = loss_fn(preds, Y)
    hessian_matvec = create_hessian_vector_product(loss, net)

    n_params = param_length(net)

    if generator is None:
        generator = gimme_new_rng()

    trace_estimates: List[float] = []

    for i in range(max_estimates):
        # Sample Rademacher probe vector (entries +/-1)
        probe = torch.randint(0, 2, (n_params,), generator=generator, device='cpu', dtype=torch.float32)
        probe = probe.mul_(2.0).sub_(1.0).to(device=device, dtype=dtype)

        Hz = hessian_matvec(probe)
        if Hz.dim() != 1 or Hz.numel() != n_params:
            raise RuntimeError("Hessian-vector product returned unexpected shape")

        trace_component = torch.dot(probe, Hz).detach().item()
        trace_estimates.append(trace_component)

        num_samples = i + 1
        if num_samples < min_estimates:
            continue

        mean_val = float(np.mean(trace_estimates))
        variance = float(np.var(trace_estimates, ddof=1)) if num_samples > 1 else 0.0

        # Avoid division by zero when the estimate is numerically zero
        if abs(mean_val) < 1e-12:
            continue

        sem = np.sqrt(variance / num_samples)
        if sem / abs(mean_val) < eps:
            break

    try:
        wandb.log({"hessian_trace_estimates": len(trace_estimates)}, commit=False)
    except Exception:
        pass

    return float(np.mean(trace_estimates))


def calculate_gradient_norm_squared_mc(net,
                                     X,
                                     Y,
                                     loss_fn,
                                     batch_size,
                                     n_estimates=1000,
                                     min_estimates=10,
                                     eps=0.005  # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96
                                     ):
    """
    Computes the Monte Carlo estimate of the expected squared norm of mini-batch gradients.
    
    This function estimates E[||∇f_B||²] where f_B is the loss on a mini-batch B,
    using Monte Carlo sampling over random mini-batches.
    
    Args:
        net (nn.Module): Neural network model
        X (Tensor): Input data tensor
        Y (Tensor): Target labels tensor  
        loss_fn (callable): Loss function that takes (outputs, targets) and returns scalar loss
        batch_size (int): Size of mini-batches to sample
        n_estimates (int, optional): Maximum number of MC estimates. Defaults to 1000.
        min_estimates (int, optional): Minimum number of estimates before checking convergence. Defaults to 10.
        eps (float, optional): Relative standard error threshold for convergence. Defaults to 0.005.
        
    Returns:
        float: Monte Carlo estimate of E[||∇f_B||²]
    """
    gradient_norm_squared_vals = []
    
    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    
    params = list(net.parameters())
    
    for i in range(n_estimates):
        # Sample random mini-batch
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]
        
        # Compute loss and gradients
        preds = net(X_batch).squeeze(dim=-1)
        loss = loss_fn(preds, Y_batch)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, params, create_graph=False)
        grads_vector = flatt(grads)
        
        # Compute squared norm of gradient
        grad_norm_squared = torch.dot(grads_vector, grads_vector).item()
        gradient_norm_squared_vals.append(grad_norm_squared)
        
        # Check convergence after minimum estimates
        if i >= min_estimates:
            mean_val = np.mean(gradient_norm_squared_vals)
            var_val = np.var(gradient_norm_squared_vals, ddof=1)
            
            # Relative standard error
            rse = np.sqrt(var_val / len(gradient_norm_squared_vals)) / abs(mean_val)
            
            if rse < eps:  # Convergence criterion
                break
    
    # Log number of estimates to wandb if available
    try:
        wandb.log({"gradient_norm_squared_mc_estimates": len(gradient_norm_squared_vals)}, commit=False)
    except:
        pass
    
    return np.mean(gradient_norm_squared_vals)


def calculate_expected_one_step_full_loss_change(net,
                                          X,
                                          Y,
                                          loss_fn,
                                          optimizer,
                                          batch_size,
                                          n_estimates=500,
                                          min_estimates=10,
                                          eps=0.005,  # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96
                                          eval_batch_size=None,  # For efficient total loss computation,
                                          use_subset_of_data: int = None # use only a subset of the dataset to calculate total loss - speeds up computations!
                                          ):
    """
    Calculate the expected one-step change in total loss using Monte Carlo estimation.
    
    This function estimates the expected change in total dataset loss when making a 
    gradient step on a randomly sampled mini-batch, then returning to the original parameters.
    
    The process for each estimate:
    1. Compute total loss before step (on entire dataset)
    2. Sample a random mini-batch for gradient computation
    3. Store current parameters
    4. Take one optimization step on the mini-batch
    5. Compute total loss after step (on entire dataset)
    6. Calculate change: (loss_after - loss_before)
    7. Restore original parameters
    
    Args:
        net (nn.Module): Neural network model
        X (Tensor): Input data tensor
        Y (Tensor): Target labels tensor
        loss_fn (callable): Loss function that takes (outputs, targets) and returns scalar loss
        optimizer (torch.optim.Optimizer): Optimizer for taking gradient steps
        batch_size (int): Size of mini-batches to sample for gradient computation
        n_estimates (int, optional): Maximum number of MC estimates. Defaults to 500.
        min_estimates (int, optional): Minimum number of estimates before checking convergence. Defaults to 10.
        eps (float, optional): Relative standard error threshold for convergence. Defaults to 0.005.
        eval_batch_size (int, optional): Batch size for total loss evaluation. If None, uses entire dataset.
        
    Returns:
        float: Monte Carlo estimate of expected total loss change
    """
    loss_changes = []
    
    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    
    # Store original parameters
    original_params = param_vector(net).detach().clone()
    
    # Compute total loss before any steps (reused for efficiency)
    # with torch.no_grad():
    if eval_batch_size is None or eval_batch_size >= len(X):
        # Evaluate on entire dataset
        preds_total_before = net(X).squeeze(dim=-1)
        total_loss_before = loss_fn(preds_total_before, Y)
    else:
        raise NotImplementedError("Batched evaluation not implemented")
    # else:
    #     # Evaluate on batches to save memory
    #     total_loss_before = 0.0
    #     n_eval_batches = (len(X) + eval_batch_size - 1) // eval_batch_size
    #     for eval_i in range(n_eval_batches):
    #         start_idx = eval_i * eval_batch_size
    #         end_idx = min((eval_i + 1) * eval_batch_size, len(X))
    #         X_eval = X[start_idx:end_idx]
    #         Y_eval = Y[start_idx:end_idx]
    #         preds_eval = net(X_eval).squeeze(dim=-1)
    #         batch_loss = loss_fn(preds_eval, Y_eval)
    #         total_loss_before += batch_loss.item() * len(X_eval)
    #     total_loss_before = total_loss_before / len(X)

    total_loss_before.backward()
    gradient_norm_squared = sum(p.grad.data.norm(2).item() ** 2 for p in net.parameters())
    eta = optimizer.param_groups[0]['lr']

    for i in range(n_estimates):
        # Sample random mini-batch for gradient step
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]
        
        # Take gradient step on mini-batch
        optimizer.zero_grad()
        preds_batch = net(X_batch).squeeze(dim=-1)
        loss_batch = loss_fn(preds_batch, Y_batch)
        loss_batch.backward()
        optimizer.step()
        
        # Compute total loss after step
        with torch.no_grad():
            if eval_batch_size is None or eval_batch_size >= len(X):
                if use_subset_of_data is not None:
                    random_idx = gimme_random_subset_idx(len(X), use_subset_of_data)

                    X_eval = X[random_idx]
                    Y_eval = Y[random_idx]
                else:
                    X_eval = X
                    Y_eval = Y

                # Evaluate on entire dataset
                preds_total_after = net(X_eval).squeeze(dim=-1)
                total_loss_after = loss_fn(preds_total_after, Y_eval)
            else:
                # Evaluate on batches to save memory
                total_loss_after = 0.0
                n_eval_batches = (len(X) + eval_batch_size - 1) // eval_batch_size
                for eval_i in range(n_eval_batches):
                    start_idx = eval_i * eval_batch_size
                    end_idx = min((eval_i + 1) * eval_batch_size, len(X))
                    X_eval = X[start_idx:end_idx]
                    Y_eval = Y[start_idx:end_idx]
                    preds_eval = net(X_eval).squeeze(dim=-1)
                    batch_loss = loss_fn(preds_eval, Y_eval)
                    total_loss_after += batch_loss.item() * len(X_eval)
                total_loss_after = total_loss_after / len(X)
            
            # Calculate change in total loss
            loss_change = total_loss_after - total_loss_before
            loss_changes.append(loss_change.item() if torch.is_tensor(loss_change) else loss_change)
        
        # Restore original parameters
        with torch.no_grad():
            param_idx = 0
            for param in net.parameters():
                param_size = param.numel()
                param.data.copy_(original_params[param_idx:param_idx + param_size].view_as(param))
                param_idx += param_size
        
        # Check convergence after minimum estimates
        if i >= min_estimates:
            mean_val = np.mean(loss_changes)
            var_val = np.var(loss_changes, ddof=1)
            
            # Relative standard error
            rse = np.sqrt(var_val / len(loss_changes)) / abs(mean_val) if mean_val != 0 else float('inf')
            
            if rse < eps:  # Convergence criterion
                break
    
    # Log number of estimates to wandb if available
    try:
        wandb.log({"one_step_total_loss_change_estimates": len(loss_changes)}, commit=False)
    except:
        pass
    
    return np.mean(loss_changes) / (eta * gradient_norm_squared)



def calculate_expected_one_step_batch_loss_change(net,
                                          X,
                                          Y,
                                          loss_fn,
                                          optimizer,
                                          batch_size,
                                          n_estimates=500,
                                          min_estimates=10,
                                          eps=0.005  # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96
                                          ):
    """
    Calculate the expected one-step change in loss using Monte Carlo estimation.
    
    This function estimates the expected relative change in loss when making a 
    gradient step on a randomly sampled batch, then returning to the original parameters.
    
    The process for each estimate:
    1. Sample a random batch
    2. Store current parameters
    3. Compute loss before step
    4. Take one optimization step
    5. Compute loss after step  
    6. Calculate relative change: (loss_after - loss_before) / loss_before
    7. Restore original parameters
    
    Args:
        net (nn.Module): Neural network model
        X (Tensor): Input data tensor
        Y (Tensor): Target labels tensor
        loss_fn (callable): Loss function that takes (outputs, targets) and returns scalar loss
        optimizer (torch.optim.Optimizer): Optimizer for taking gradient steps
        batch_size (int): Size of mini-batches to sample
        n_estimates (int, optional): Maximum number of MC estimates. Defaults to 500.
        min_estimates (int, optional): Minimum number of estimates before checking convergence. Defaults to 10.
        eps (float, optional): Relative standard error threshold for convergence. Defaults to 0.005.
        
    Returns:
        float: Monte Carlo estimate of expected relative one-step loss change
    """
    loss_changes = []
    
    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    
    # Store original parameters
    original_params = param_vector(net).detach().clone()
    
    for i in range(n_estimates):
        # Sample random mini-batch
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]
        
        # Compute loss before step
        optimizer.zero_grad()
        preds_before = net(X_batch).squeeze(dim=-1)
        loss_before = loss_fn(preds_before, Y_batch)
        
        # Take gradient step
        loss_before.backward()
        optimizer.step()
        
        # Compute loss after step (on the same batch)
        with torch.no_grad():
            preds_after = net(X_batch).squeeze(dim=-1)
            loss_after = loss_fn(preds_after, Y_batch)
            
            # Calculate relative change in loss
            relative_change = (loss_after - loss_before) #/ loss_before
            loss_changes.append(relative_change.item())
        
        # Restore original parameters
        current_params = param_vector(net)
        with torch.no_grad():
            param_idx = 0
            for param in net.parameters():
                param_size = param.numel()
                param.data.copy_(original_params[param_idx:param_idx + param_size].view_as(param))
                param_idx += param_size
        
        # Check convergence after minimum estimates
        if i >= min_estimates:
            mean_val = np.mean(loss_changes)
            var_val = np.var(loss_changes, ddof=1)
            
            # Relative standard error
            rse = np.sqrt(var_val / len(loss_changes)) / abs(mean_val) if mean_val != 0 else float('inf')
            
            if rse < eps:  # Convergence criterion
                break
    
    # Log number of estimates to wandb if available
    try:
        wandb.log({"one_step_loss_change_estimates": len(loss_changes)}, commit=False)
    except:
        pass
    
    return np.mean(loss_changes)


################################################################################
#                                                                              #
#                        GAUSS–NEWTON (=FIM) MATRIX STUFF                      #
#                                                                              #
################################################################################

def calculate_all_net_grads(net, X):

    gradients = []
    params = list(net.parameters())

    for x in X:
        y = net(x.unsqueeze(0))
        # compute gradient
        grads = torch.autograd.grad(y, params)
        grads_vector = flatt(grads).detach()
        gradients.append(grads_vector)
    
    G = T.stack(gradients)
    del gradients
    return G



def create_ntk(net, X):
    params = list(net.parameters())

    gradients = []

    for x in X:
        y = net(x.unsqueeze(0))
        # compute gradient
        grads = torch.autograd.grad(y, params)
        grads_vector = flatt(grads).detach()
        gradients.append(grads_vector)
    
    G = T.stack(gradients)

    ntk = G @ G.T
    del G
    # f = lambda v: G.T @ (G @ v) / len(X)

    return ntk


def compute_fisher_eigenvalues(net, X):
    '''
    The trick here is that instead of computing the fisher information matrix, we compute the NTK
    They have the same eigenvalues, but NTK is size n_samples x n_samples, while FIM is size n_params x n_params
    '''

    ntk = create_ntk(net, X)
    # size = param_length(net)

    # device = next(net.parameters()).device
    # eigenval = compute_eigenvalues(operator, size, device, iterations=iterations, epsilon=epsilon)

    eigenval = T.lobpcg(ntk, k=1)
    eigenval = 2/len(X) * eigenval[0]
    
    return eigenval




################################################################################
#                                                                              #
#                                LAMBDA^b_MAX                                  #
#                                                                              #
################################################################################


def calculate_averaged_lambdamax(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                              max_hessian_iters = 1000,
                              hessian_tolerance = 1e-3,
                              batched = None,
                              compute_gHg: bool = False,
                              eigenvector_cache = None
                              ): 
    
    
    sharpnesses = []

    if compute_gHg:
        gHg_values = []
    
    if batch_size is None:
        batch_size = len(X)
    

    for i in range(n_estimates):
        shuffle = T.randperm(len(X))
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]


        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        sharpness = compute_eigenvalues(loss, 
                        net,
                        max_iterations=max_hessian_iters,
                        reltol=hessian_tolerance,
                        )
        if compute_gHg:
            sharpness, gHg = sharpness
            gHg = gHg.item()
            gHg_values.append(gHg)
        
        sharpness = sharpness.item()
        
        sharpnesses.append(sharpness)

        if batch_size >= len(X):
            break

        if len(sharpnesses) > min_estimates:
            mean = np.mean(sharpnesses)
            sem = np.std(sharpnesses) / np.sqrt(len(sharpnesses))

            if sem / mean < tolerance:
                break
    
    if compute_gHg:
        return sharpnesses, gHg_values
    return sharpnesses


################################################################################
#                                                                              #
#                                NEW MEMORIZATION METRICS                      #
#                                                                              #
################################################################################
def compute_train_test_gap_from_tensors(net, X_train, Y_train, X_test, Y_test) -> dict:
    with torch.no_grad():
        train_logits = net(X_train).squeeze(dim=-1)  # Add this squeeze
        test_logits  = net(X_test).squeeze(dim=-1)   # Add this squeeze

    train_acc = float(calculate_accuracy(train_logits, Y_train))
    test_acc = float(calculate_accuracy(test_logits,  Y_test))
    gap = train_acc - test_acc
    
    try:
        wandb.log({"memorization/train_acc": train_acc, 
                    "memorization/test_acc": test_acc, "train_test_gap": gap}, commit=False)
    except Exception:
        pass
   
    return {"train_acc": train_acc, "test_acc": test_acc, "gap": gap}


def compute_per_example_losses(net, X, Y, loss_fn) -> torch.Tensor:
    """
    Compute per-example losses for a batch, preserving gradients when supported by the loss_fn.
    """
    device = next(net.parameters()).device
    X = X.to(device)
    Y = Y.to(device)

    logits = net(X).squeeze(dim=-1)

    losses = None

    # Preferred: explicitly request per-sample losses if the signature supports it
    try:
        losses = loss_fn(logits, Y, reduction="none")
    except (TypeError, ValueError):
        losses = None

    # Fallback: temporarily flip the reduction attribute if present
    if losses is None and hasattr(loss_fn, "reduction"):
        original_reduction = loss_fn.reduction
        try:
            loss_fn.reduction = "none"
            losses = loss_fn(logits, Y)
        except Exception:
            losses = None
        finally:
            loss_fn.reduction = original_reduction

    # Last resort: loop over samples
    if losses is None:
        losses_list = []
        for x_i, y_i in zip(X, Y):
            loss_i = loss_fn(
                net(x_i.unsqueeze(0)).squeeze(dim=-1),
                y_i.unsqueeze(0),
            )
            loss_i = loss_i.view(-1)[0] if loss_i.numel() == 1 else loss_i
            losses_list.append(loss_i)
        losses = torch.stack(losses_list)

    return losses.view(-1)


def compute_outlier_vs_bulk_stats_hessian(
    net,
    X_train,
    Y_train,
    loss_fn,
    optimizer,
    frac: float = 0.05,
) -> dict:
    """
    Identify outlier examples by alignment with the top Hessian eigenvector and
    report summary statistics for outliers vs the remaining bulk.
    """
    device = next(net.parameters()).device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)

    preds_full = net(X_train).squeeze(dim=-1)
    loss_full = loss_fn(preds_full, Y_train)

    lambdamax, v = compute_eigenvalues(
        loss_full,
        net,
        k=1,
        max_iterations=100,
        reltol=1e-2,
        eigenvector_cache=None,
        return_eigenvectors=True,
    )
    v = v.detach()

    grads = calculate_all_the_grads(
        net,
        X_train,
        Y_train,
        loss_fn,
        optimizer,
        storage_device="cpu",
    )
    grads = grads.to(device)

    scores = grads @ v
    abs_scores = scores.abs()
    N = abs_scores.shape[0]
    k = max(1, int(frac * N))
    threshold = torch.topk(abs_scores, k=k, largest=True).values.min()
    outlier_mask = abs_scores >= threshold
    bulk_mask = ~outlier_mask

    if outlier_mask.sum().item() == 0 or bulk_mask.sum().item() == 0:
        return {}

    losses = compute_per_example_losses(net, X_train, Y_train, loss_fn)

    with torch.no_grad():
        logits = net(X_train).squeeze(dim=-1)
        train_acc = calculate_accuracy(logits, Y_train)

    out_idx = outlier_mask.nonzero(as_tuple=False).view(-1)
    bulk_idx = bulk_mask.nonzero(as_tuple=False).view(-1)

    out_losses = losses[out_idx]
    bulk_losses = losses[bulk_idx]

    with torch.no_grad():
        out_acc = calculate_accuracy(logits[out_idx], Y_train[out_idx])
        bulk_acc = calculate_accuracy(logits[bulk_idx], Y_train[bulk_idx])

    metrics = {
        "outliers/fraction": float(frac),
        "outliers/count": int(len(out_idx)),
        "outliers/mean_loss": float(out_losses.mean().item()),
        "outliers/mean_alignment": float(abs_scores[out_idx].mean().item()),
        "outliers/accuracy": float(out_acc),
        "bulk/mean_loss": float(bulk_losses.mean().item()),
        "bulk/mean_alignment": float(abs_scores[bulk_idx].mean().item()),
        "bulk/accuracy": float(bulk_acc),
        "train/accuracy": float(train_acc),
        "hessian/lambdamax": float(lambdamax.item()),
    }

    try:
        wandb.log(
            {f"memorization_hessian_outliers/{key}": val for key, val in metrics.items()},
            commit=False,
        )
    except Exception:
        pass

    return metrics


@torch.no_grad()
def extract_feature_matrix(
    net: nn.Module,
    inputs: torch.Tensor,
    batch_size: int = 512,
    flatten_outputs: bool = True,
    device: Optional[torch.device] = None,
):
    """
    Run `net` over `inputs` and collect a matrix of features/logits.

    Args:
        net: Trained network.
        inputs: Tensor containing the dataset to be embedded.
        batch_size: Mini-batch size used during the forward pass.
        flatten_outputs: If True, flattens non-vector outputs to 2D.
        device: Optional override for the computation device.

    Returns:
        Tensor of shape (N, D) on CPU containing the feature vectors.
    """
    if inputs.ndim < 2:
        raise ValueError("inputs must have at least 2 dimensions (batch, ...)")

    target_device = device or next(net.parameters()).device
    was_training = net.training
    net.eval()

    features = []
    total = inputs.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = inputs[start:end].to(target_device)
        outputs = net(batch)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(1)
        if flatten_outputs and outputs.ndim > 2:
            outputs = outputs.view(outputs.shape[0], -1)
        features.append(outputs.detach().to('cpu', copy=True))

    if was_training:
        net.train()

    return torch.cat(features, dim=0)


def identify_knn_outliers_by_neighbor_mix(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    k_neighbors: int = 32,
    top_k_per_class: int = 5,
    balance_target: float = 0.5,
    chunk_size: int = 1024,
    normalize: bool = True,
    return_neighbor_indices: bool = False,
):
    """
    Identify samples whose k-NN neighborhood is closest to a 50/50 class split.

    Args:
        features: Tensor of shape (N, D) containing feature vectors (e.g., logits).
        labels: Tensor of shape (N,) or one-hot encoded labels.
        k_neighbors: Number of nearest neighbors to inspect for each sample.
        top_k_per_class: Number of outlier candidates to keep per class.
        balance_target: Desired ratio for same-class neighbors (0.5 = perfect balance).
        chunk_size: How many query samples to process at once when building the k-NN graph.
        normalize: If True, L2-normalize feature vectors before computing distances.
        return_neighbor_indices: If True, include neighbor indices in the output.

    Returns:
        Dictionary summarizing outliers per class together with k-NN mix statistics.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor of shape (N, D)")

    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1")

    if k_neighbors >= features.shape[0]:
        raise ValueError("k_neighbors must be smaller than the number of samples")

    feats = features.detach().to(torch.float32)
    if normalize:
        feats = torch.nn.functional.normalize(feats, dim=1)
    if feats.device.type != 'cpu':
        feats = feats.cpu()

    labels_cpu = labels.detach().cpu()
    if labels_cpu.ndim > 1:
        labels_cpu = torch.argmax(labels_cpu, dim=1)
    labels_cpu = labels_cpu.to(torch.long)

    class_values, mapped_labels = torch.unique(labels_cpu, sorted=True, return_inverse=True)
    num_classes = class_values.numel()
    if num_classes == 0:
        return {}

    n_samples = feats.shape[0]
    chunk_size = max(1, min(chunk_size, n_samples))

    knn_indices = torch.empty((n_samples, k_neighbors), dtype=torch.long)
    knn_distances = torch.empty((n_samples, k_neighbors), dtype=feats.dtype)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        query = feats[start:end]
        dists = torch.cdist(query, feats, p=2)
        rows = torch.arange(end - start, device=dists.device)
        cols = torch.arange(start, end, device=dists.device)
        dists[rows, cols] = float('inf')
        dist_vals, idx = torch.topk(dists, k=k_neighbors, dim=1, largest=False)
        knn_indices[start:end] = idx
        knn_distances[start:end] = dist_vals

    neighbor_labels = mapped_labels[knn_indices]
    neighbor_counts = torch.nn.functional.one_hot(
        neighbor_labels, num_classes=num_classes
    ).sum(dim=1)

    same_class_counts = neighbor_counts[torch.arange(n_samples), mapped_labels]
    same_class_ratio = same_class_counts.to(torch.float32) / float(k_neighbors)
    balance_deviation = (same_class_ratio - balance_target).abs()

    neighbor_probs = neighbor_counts.to(torch.float32) / float(k_neighbors)
    entropy = -(neighbor_probs.clamp_min(1e-12) * neighbor_probs.clamp_min(1e-12).log()).sum(dim=1)

    if num_classes >= 2:
        top_two = torch.topk(neighbor_probs, k=2, dim=1).values
        mix_gap = (top_two[:, 0] - top_two[:, 1]).abs()
    else:
        mix_gap = torch.zeros_like(same_class_ratio)

    per_class_results = {}
    for class_idx, class_value in enumerate(class_values):
        class_mask = (mapped_labels == class_idx)
        class_indices = class_mask.nonzero(as_tuple=False).view(-1)
        if class_indices.numel() == 0:
            continue

        class_scores = balance_deviation[class_indices]
        keep = min(top_k_per_class, class_scores.numel())
        if keep == 0:
            continue
        _, top_local_idx = torch.topk(class_scores, k=keep, largest=False)
        selected_indices = class_indices[top_local_idx]

        entries = []
        for idx in selected_indices.tolist():
            entry = {
                "dataset_index": int(idx),
                "same_class_ratio": float(same_class_ratio[idx].item()),
                "balance_deviation": float(balance_deviation[idx].item()),
                "neighbor_entropy": float(entropy[idx].item()),
                "top_two_gap": float(mix_gap[idx].item()),
                "neighbor_class_counts": {
                    int(class_values[j].item()): int(neighbor_counts[idx, j].item())
                    for j in range(num_classes)
                },
                "avg_neighbor_distance": float(knn_distances[idx].mean().item()),
            }
            if return_neighbor_indices:
                entry["neighbor_indices"] = knn_indices[idx].tolist()
            entries.append(entry)

        per_class_results[int(class_value.item())] = entries

    return {
        "k_neighbors": int(k_neighbors),
        "top_k_per_class": int(top_k_per_class),
        "balance_target": float(balance_target),
        "num_samples": int(n_samples),
        "class_ids": [int(val.item()) for val in class_values],
        "outliers": per_class_results,
    }


def select_dataset_subset(
    X: torch.Tensor,
    Y: torch.Tensor,
    indices: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Slice tensors X/Y along the first dimension according to `indices`.

    Args:
        X: Tensor of inputs.
        Y: Tensor of targets.
        indices: 1-D tensor/list of indices. If None, returns (X, Y).

    Returns:
        (X_subset, Y_subset)
    """
    if indices is None:
        return X, Y

    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, dtype=torch.long, device=X.device)
    else:
        indices = indices.to(dtype=torch.long, device=X.device)

    return X.index_select(0, indices), Y.index_select(0, indices)


def _compute_metrics_on_subset_data(
    net: nn.Module,
    loss_fn,
    X_subset: torch.Tensor,
    Y_subset: torch.Tensor,
    metrics: List[str],
    *,
    eigenvector_cache: Optional[EigenvectorCache],
    num_eigenvalues: int,
    use_power_iteration: bool,
    metric_kwargs: dict,
) -> dict:
    if X_subset.numel() == 0:
        raise ValueError("Subset is empty; cannot compute metrics.")

    device = next(net.parameters()).device
    X_subset = X_subset.to(device)
    Y_subset = Y_subset.to(device)

    was_training = net.training
    net.eval()

    results = {}

    preds = net(X_subset).squeeze(dim=-1)
    if (
        isinstance(loss_fn, nn.modules.loss._Loss)
        and loss_fn.__class__.__name__ == "SquaredLoss"
        and Y_subset.ndim == 1
        and preds.ndim == 2
    ):
        # Convert class-index labels to one-hot targets for MSE-style losses.
        Y_subset = F.one_hot(Y_subset.long(), num_classes=preds.shape[1]).float()
    loss_value = loss_fn(preds, Y_subset)
    subset_size = X_subset.shape[0]

    def _maybe_add_batch_sharpness(metric_name: str, *, expectation_inside_default: bool):
        if metric_name not in metrics:
            return

        cfg = metric_kwargs.get(metric_name)
        if cfg is None and metric_name != "batch_sharpness":
            cfg = metric_kwargs.get("batch_sharpness", {})
        if cfg is None:
            cfg = {}

        bs_value = calculate_averaged_grad_H_grad(
            net=net,
            X=X_subset,
            Y=Y_subset,
            loss_fn=loss_fn,
            batch_size=cfg.get("batch_size", subset_size),
            n_estimates=cfg.get("n_estimates", 1000),
            min_estimates=cfg.get("min_estimates", 20),
            eps=cfg.get("eps", 0.005),
            expectation_inside=cfg.get("expectation_inside", expectation_inside_default),
            with_replacement=cfg.get("with_replacement", False),
            return_confidence_interval=cfg.get("return_confidence_interval", False),
            confidence_level=cfg.get("confidence_level", 0.95),
        )

        if isinstance(bs_value, dict):
            mean_val = bs_value.get("mean")
            results[metric_name] = float(mean_val) if mean_val is not None else float("nan")
            ci = bs_value.get("ci")
            if ci is not None and len(ci) == 2:
                results[f"{metric_name}_ci_low"] = float(ci[0])
                results[f"{metric_name}_ci_high"] = float(ci[1])
            stderr = bs_value.get("stderr")
            if stderr is not None:
                results[f"{metric_name}_stderr"] = float(stderr)
            conf = bs_value.get("confidence_level")
            if conf is not None:
                results[f"{metric_name}_confidence_level"] = float(conf)
            num_samples = bs_value.get("num_samples")
            if num_samples is not None:
                results[f"{metric_name}_num_samples"] = int(num_samples)
        else:
            results[metric_name] = float(bs_value)

    if "full_loss" in metrics:
        results["full_loss"] = float(loss_value.item())

    if "accuracy" in metrics:
        with torch.no_grad():
            results["accuracy"] = float(calculate_accuracy(preds, Y_subset))

    if "lambda_max" in metrics or "all_eigenvalues" in metrics:
        eig_kwargs = dict(metric_kwargs.get("lambda_max", {}))
        k = eig_kwargs.pop("k", num_eigenvalues)
        return_eigenvectors = eig_kwargs.pop("return_eigenvectors", False)
        eigenvalues = compute_eigenvalues(
            loss_value,
            net,
            k=k,
            eigenvector_cache=eigenvector_cache,
            use_power_iteration=eig_kwargs.pop("use_power_iteration", use_power_iteration),
            **eig_kwargs,
        )
        if return_eigenvectors:
            eigenvalues, eigenvectors = eigenvalues
            results["all_eigenvalues"] = eigenvalues.detach().cpu().tolist()
            results["lambda_max"] = float(eigenvalues[0].item())
            results["eigenvectors"] = eigenvectors
        else:
            if torch.is_tensor(eigenvalues):
                if eigenvalues.numel() == 1:
                    results["lambda_max"] = float(eigenvalues.item())
                else:
                    results["all_eigenvalues"] = eigenvalues.detach().cpu().tolist()
                    results["lambda_max"] = float(eigenvalues[0].item())
            elif isinstance(eigenvalues, tuple):
                vals = eigenvalues[0]
                results["all_eigenvalues"] = vals.detach().cpu().tolist()
                results["lambda_max"] = float(vals[0].item())

    if "grad_hessian_grad" in metrics:
        results["grad_hessian_grad"] = float(compute_grad_H_grad(loss_value, net).item())

    if "grad_vmax_cos2" in metrics or "grad_norm" in metrics:
        if (eigenvector_cache is not None
                and hasattr(eigenvector_cache, 'eigenvectors')
                and len(eigenvector_cache.eigenvectors) > 0):
    
            params = [p for p in net.parameters() if p.requires_grad]
            net.zero_grad()
            preds = net(X_subset).squeeze(dim=-1)
            loss_for_grad = loss_fn(preds, Y_subset)
            grad_list = torch.autograd.grad(
                loss_for_grad, params, create_graph=False, retain_graph=False
            )
            grad_flat = torch.cat([g.reshape(-1) for g in grad_list]).detach()
    
            g_norm = grad_flat.norm()
    
            if "grad_norm" in metrics:
                results["grad_norm"] = float(g_norm.item())
    
            if "grad_vmax_cos2" in metrics:
                v_max = eigenvector_cache.eigenvectors[0].to(grad_flat.device)
                v_max = v_max / (v_max.norm() + 1e-12)
                if g_norm > 1e-12:
                    cos2 = ((grad_flat / g_norm) @ v_max).pow(2).item()
                else:
                    cos2 = float('nan')
                results["grad_vmax_cos2"] = cos2
        else:
            if "grad_norm" in metrics:
                results["grad_norm"] = float('nan')
            if "grad_vmax_cos2" in metrics:
                results["grad_vmax_cos2"] = float('nan')
        
    if "per_example_loss_mean" in metrics or "per_example_loss_std" in metrics:
        with torch.no_grad():
            per_example_losses = compute_per_example_losses(net, X_subset, Y_subset, loss_fn)
        if "per_example_loss_mean" in metrics:
            results["per_example_loss_mean"] = float(per_example_losses.mean().item())
        if "per_example_loss_std" in metrics:
            results["per_example_loss_std"] = float(per_example_losses.std(unbiased=False).item())

    _maybe_add_batch_sharpness("batch_sharpness", expectation_inside_default=False)
    _maybe_add_batch_sharpness("batch_sharpness_exp_inside", expectation_inside_default=True)

    if "gradient_norm_squared" in metrics:
        gn_kwargs = metric_kwargs.get("gradient_norm_squared", {})
        results["gradient_norm_squared"] = float(
            calculate_gradient_norm_squared_mc(
                net=net,
                X=X_subset,
                Y=Y_subset,
                loss_fn=loss_fn,
                batch_size=gn_kwargs.get("batch_size", len(X_subset)),
                n_estimates=gn_kwargs.get("n_estimates", 100),
                min_estimates=gn_kwargs.get("min_estimates", 10),
                eps=gn_kwargs.get("eps", 0.01),
            )
        )

    if "gni" in metrics:
        gni_kwargs = metric_kwargs.get("gni", {})
        results["gni"] = float(
            calculate_gni(
                net=net,
                X=X_subset,
                Y=Y_subset,
                loss_fn=loss_fn,
                batch_size=gni_kwargs.get("batch_size", len(X_subset)),
                n_estimates=gni_kwargs.get("n_estimates", 200),
                min_estimates=gni_kwargs.get("min_estimates", 10),
                tolerance=gni_kwargs.get("tolerance", 0.01),
                use_subset_of_data=gni_kwargs.get("use_subset_of_data", None),
            )
        )

    if "hessian_trace" in metrics:
        trace_kwargs = metric_kwargs.get("hessian_trace", {})
        results["hessian_trace"] = float(
            estimate_hessian_trace(
                net=net,
                X=X_subset,
                Y=Y_subset,
                loss_fn=loss_fn,
                max_estimates=trace_kwargs.get("max_estimates", 256),
                min_estimates=trace_kwargs.get("min_estimates", 10),
                eps=trace_kwargs.get("eps", 0.01),
                probe_type=trace_kwargs.get("probe_type", "rademacher"),
            )
        )

    if was_training:
        net.train()

    return results


def compute_subset_metrics(
    net: nn.Module,
    loss_fn,
    X: torch.Tensor,
    Y: torch.Tensor,
    indices,
    metrics: Optional[List[str]] = None,
    *,
    eigenvector_cache: Optional[EigenvectorCache] = None,
    num_eigenvalues: int = 1,
    use_power_iteration: bool = False,
    metric_kwargs: Optional[dict] = None,
) -> dict:
    """
    Compute measurement-runner style metrics restricted to a subset of examples.
    """
    if metrics is None:
        metrics = ["full_loss", "accuracy", "lambda_max"]
    if metric_kwargs is None:
        metric_kwargs = {}

    X_subset, Y_subset = select_dataset_subset(X, Y, indices)
    return _compute_metrics_on_subset_data(
        net=net,
        loss_fn=loss_fn,
        X_subset=X_subset,
        Y_subset=Y_subset,
        metrics=metrics,
        eigenvector_cache=eigenvector_cache,
        num_eigenvalues=num_eigenvalues,
        use_power_iteration=use_power_iteration,
        metric_kwargs=metric_kwargs,
    )


def compute_subset_metrics_from_tensors(
    net: nn.Module,
    loss_fn,
    X_subset: torch.Tensor,
    Y_subset: torch.Tensor,
    metrics: Optional[List[str]] = None,
    *,
    eigenvector_cache: Optional[EigenvectorCache] = None,
    num_eigenvalues: int = 1,
    use_power_iteration: bool = False,
    metric_kwargs: Optional[dict] = None,
) -> dict:
    """
    Compute subset metrics directly on provided tensors (no indexing into a parent dataset).
    """
    if metrics is None:
        metrics = ["full_loss", "accuracy", "lambda_max"]
    if metric_kwargs is None:
        metric_kwargs = {}

    return _compute_metrics_on_subset_data(
        net=net,
        loss_fn=loss_fn,
        X_subset=X_subset,
        Y_subset=Y_subset,
        metrics=metrics,
        eigenvector_cache=eigenvector_cache,
        num_eigenvalues=num_eigenvalues,
        use_power_iteration=use_power_iteration,
        metric_kwargs=metric_kwargs,
    )
