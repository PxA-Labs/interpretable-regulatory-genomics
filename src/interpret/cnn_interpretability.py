import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_filters_to_pwms(model, softmax_temp: float = 1.0) -> list:
    """
    Extract the weights of the first convolutional layer and convert them to
    Position Weight Matrices (PWMs) using Softmax over the nucleotide dimension.
    
    Parameters:
    -----------
    model : nn.Module or PyTorchModelWrapper
        Trained model containing a conv1 layer.
    softmax_temp : float, default 1.0
        Temperature parameter for the Softmax normalization. Lower temperatures
        accentuate dominant nucleotides, higher temperatures make them more uniform.
        
    Returns:
    --------
    list of np.ndarray
        List of PWMs of shape (4, kernel_size). Each column sums to 1.0.
    """
    # Extract underlying PyTorch module if wrapped
    if hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model
        
    # Check if model has conv1
    if not hasattr(pytorch_model, 'conv1'):
        raise ValueError("Model does not have a 'conv1' first convolutional layer.")
        
    conv_layer = pytorch_model.conv1
    # Weights shape: (out_channels, in_channels=4, kernel_size)
    weights = conv_layer.weight.detach().cpu().numpy()
    
    pwms = []
    for f in range(weights.shape[0]):
        filter_weights = weights[f] # Shape: (4, kernel_size)
        # Apply Softmax across the nucleotide dimension (dim 0)
        # Exponentiate
        exp_w = np.exp(filter_weights / softmax_temp)
        pwm = exp_w / np.sum(exp_w, axis=0, keepdims=True)
        pwms.append(pwm)
        
    return pwms

def plot_filters_grid(pwms: list, output_plot_path: str, max_display: int = 16):
    """
    Plot a grid of sequence logos representing the learned filters.
    Utilizes logomaker if available, falling back to a heatmap if not.
    """
    n_display = min(max_display, len(pwms))
    grid_size = int(np.ceil(np.sqrt(n_display)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3.5, grid_size * 2))
    axes = axes.flatten()
    
    # Try importing logomaker
    try:
        import logomaker
        has_logomaker = True
    except ImportError:
        has_logomaker = False
        print("Notice: logomaker library not installed. Falling back to heatmap visualization for filters.")
        
    for i in range(len(axes)):
        ax = axes[i]
        if i < n_display:
            pwm = pwms[i] # Shape: (4, kernel_size)
            # Transpose to shape (kernel_size, 4)
            pwm_df = pd.DataFrame(pwm.T, columns=['A', 'C', 'G', 'T'])
            
            if has_logomaker:
                # Calculate information content / height (simple approximation using Shannon entropy)
                # height = probability * (2 - entropy)
                entropy = -np.sum(pwm * np.log2(pwm + 1e-9), axis=0)
                heights = pwm_df.multiply(2.0 - entropy, axis=0)
                logomaker.Logo(heights, ax=ax, color_scheme='classic', width=0.8)
                ax.set_title(f"Filter {i}", fontsize=10)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xticks([])
            else:
                im = ax.imshow(pwm, cmap='Blues', aspect='auto')
                ax.set_yticks([0, 1, 2, 3])
                ax.set_yticklabels(['A', 'C', 'G', 'T'], fontsize=8)
                ax.set_title(f"Filter {i}", fontsize=10)
                ax.set_xticks([])
        else:
            ax.axis('off')
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved filter visualization grid to {output_plot_path}")

def save_pwms_to_meme(pwms: list, output_path: str, bg_freqs: dict = None):
    """
    Save extracted PWMs in standard MEME format to enable downstream scanning
    using motif matching tools like Tomtom (MEME Suite).
    """
    if bg_freqs is None:
        bg_freqs = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
        
    print(f"Writing {len(pwms)} filters to MEME file: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write(f"A {bg_freqs['A']:.4f} C {bg_freqs['C']:.4f} G {bg_freqs['G']:.4f} T {bg_freqs['T']:.4f}\n\n")
        
        for idx, pwm in enumerate(pwms):
            # Shape: (4, kernel_size)
            kernel_size = pwm.shape[1]
            f.write(f"MOTIF filter_{idx} filter_{idx}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {kernel_size} nsites= 100 E= 0\n")
            
            # Print columns
            for col in range(kernel_size):
                f.write(f"  {pwm[0, col]:.6f}  {pwm[1, col]:.6f}  {pwm[2, col]:.6f}  {pwm[3, col]:.6f}\n")
            f.write("\n")

def compute_attribution(model, input_tensor, method: str = 'saliency', target=None) -> np.ndarray:
    """
    Calculate per-nucleotide attributions using Saliency or Integrated Gradients.
    Uses Captum if installed, otherwise falls back to custom gradient computations.
    
    Parameters:
    -----------
    model : nn.Module or PyTorchModelWrapper
        The model to explain.
    input_tensor : torch.Tensor
        Batch of input sequences of shape (batch_size, 4, 1000).
    method : str, default 'saliency'
        'saliency' or 'integrated_gradients'.
    target : int, optional
        Target output node index (not used for single-output binary classifiers).
        
    Returns:
    --------
    np.ndarray
        Attribution values of shape (batch_size, 4, 1000).
    """
    # Extract underlying PyTorch module if wrapped
    if hasattr(model, 'model'):
        pytorch_model = model.model
        device = model.device
    else:
        pytorch_model = model
        device = next(pytorch_model.parameters()).device
        
    pytorch_model.eval()
    x = input_tensor.to(device).clone().detach().requires_grad_(True)
    
    # Try importing Captum
    try:
        from captum.attr import Saliency, IntegratedGradients
        has_captum = True
    except ImportError:
        has_captum = False
        
    if has_captum:
        if method.lower() == 'saliency':
            sal = Saliency(pytorch_model)
            # For Captum, input requires_grad is handled inside
            attributions = sal.attribute(x)
        elif method.lower() == 'integrated_gradients':
            ig = IntegratedGradients(pytorch_model)
            baseline = torch.zeros_like(x)
            attributions = ig.attribute(x, baselines=baseline, n_steps=20)
        else:
            raise ValueError(f"Unknown method: {method}")
        return attributions.detach().cpu().numpy()
        
    else:
        # Fallback to direct PyTorch autograd gradients
        print("Notice: Captum not installed. Computing attributions via standard PyTorch autograd.")
        
        # Enable gradient tracking
        logits = pytorch_model(x)
        
        # Backpropagate gradients of the logit sum
        pytorch_model.zero_grad()
        logits.sum().backward()
        
        grads = x.grad.detach().cpu().numpy()
        
        if method.lower() == 'saliency':
            # Saliency maps: Gradient
            return grads
        elif method.lower() == 'integrated_gradients':
            # Approximation of Integrated Gradients (using a simple linear interpolation)
            # 10 steps Riemann sum approximation
            steps = 10
            integrated_grads = np.zeros_like(grads)
            x_np = x.detach().cpu().numpy()
            
            for step in range(1, steps + 1):
                alpha = step / steps
                x_interpolated = torch.tensor(alpha * x_np, dtype=torch.float32, device=device, requires_grad=True)
                outputs = pytorch_model(x_interpolated)
                pytorch_model.zero_grad()
                outputs.sum().backward()
                integrated_grads += x_interpolated.grad.detach().cpu().numpy()
                
            integrated_grads = integrated_grads / steps * x_np
            return integrated_grads
        else:
            raise ValueError(f"Unknown method: {method}")

def plot_sequence_attribution(sequence_onehot: np.ndarray, attributions: np.ndarray, 
                              output_path: str, start_pos: int = 400, end_pos: int = 600):
    """
    Plot per-nucleotide sequence logo attribution (Gradient x Input).
    
    Parameters:
    -----------
    sequence_onehot : np.ndarray
        One-hot encoded DNA sequence of shape (4, 1000).
    attributions : np.ndarray
        Calculated attributions of shape (4, 1000).
    output_path : str
        Where to save the plot.
    start_pos : int, default 400
        Beginning index of window to visualize (e.g. around the center).
    end_pos : int, default 600
        Ending index of window.
    """
    # Gradient x Input representation
    grad_x_input = attributions * sequence_onehot # Shape: (4, 1000)
    
    # Slice the visualization window
    window_seq = sequence_onehot[:, start_pos:end_pos]
    window_attr = grad_x_input[:, start_pos:end_pos]
    length = end_pos - start_pos
    
    # Try importing logomaker
    try:
        import logomaker
        has_logomaker = True
    except ImportError:
        has_logomaker = False
        
    fig, ax = plt.subplots(figsize=(15, 3))
    
    if has_logomaker:
        # Construct DataFrame
        df = pd.DataFrame(window_attr.T, columns=['A', 'C', 'G', 'T'])
        df.index = range(start_pos, end_pos)
        
        # Plot logo
        logo = logomaker.Logo(df, ax=ax, color_scheme='classic', width=0.8)
        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        ax.set_ylabel("Attribution (Grad x Input)", fontsize=11)
        ax.set_xlabel("Genomic Position (bp)", fontsize=11)
    else:
        # Fallback heatmap representation
        bases = ['A', 'C', 'G', 'T']
        im = ax.imshow(window_attr, cmap='RdBu', aspect='auto', interpolation='nearest',
                       vmin=-np.max(np.abs(window_attr)), vmax=np.max(np.abs(window_attr)))
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(bases)
        ax.set_xticks(range(0, length, 20))
        ax.set_xticklabels(range(start_pos, end_pos, 20))
        plt.colorbar(im, ax=ax, label="Attribution")
        
    plt.title(f"Sequence Attribution Map (Positions {start_pos} - {end_pos})", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sequence attribution logo plot to {output_path}")
