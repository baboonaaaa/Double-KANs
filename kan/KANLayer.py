import torch
import torch.nn as nn
import torch.nn.functional as F
from kan.spline import *


class KANLayer(nn.Module):
    """
    Double-KAN Layer Implementation
    
    This is our improved version of the Kolmogorov-Arnold Network layer that introduces
    a novel dual-coefficient normalization approach for enhanced representation learning.
    
    Key Innovation:
    - Uses two separate coefficient sets (coef1, coef2) with adaptive normalization
    - Implements y = coef1 / (softplus(coef2^2) + ε) for improved stability
    - Maintains the original KAN philosophy while enhancing performance
    
    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        num: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
        W_prime_I: 2D torch.tensor
            composite coefficients w'_i for numerator in Eq(11)
        W_prime_K: 2D torch.tensor
            composite coefficients w'_k for denominator in Eq(11)
        scale_base: float
            magnitude of the residual function b(x)
        scale_sp: float
            magnitude of the spline function
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions
        device: str
            device
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, 
                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0, 
                 base_fun=torch.nn.SiLU(), grid_range=[-1, 1], 
                 sp_trainable=True, sb_trainable=True, device='cpu', sparse_init=False):
        """
        Initialize a Double-KAN Layer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 3.
            out_dim : int
                output dimension. Default: 2.
            num : int
                the number of grid intervals. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.5.
            scale_base_mu : float
                mean for base function scale initialization. Default: 0.0.
            scale_base_sigma : float
                std for base function scale initialization. Default: 1.0.
            scale_sp : float
                the scale of the spline function. Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU().
            grid_range : list
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device. Default: 'cpu'.
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied. Default: False.
        """
        super(KANLayer, self).__init__()
        
        # Layer dimensions
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # Initialize grid
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(self.in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        
        # Initialize noise for coefficients
        noises = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 1 / 2) * noise_scale / num

        # Initialize dual coefficients corresponding to w'_i and w'_k in Eq(11)
        self.WI = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))
        self.WK = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))
        
        # Initialize mask
        if sparse_init:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim) * 0.0).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)

        # Initialize scales
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / (in_dim ** 0.5) + \
                                             scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) / (in_dim ** 0.5)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(
            torch.ones(in_dim, out_dim) * scale_sp / (in_dim ** 0.5) * self.mask).requires_grad_(sp_trainable)
        
        self.base_fun = base_fun
        self.to(device)

    def to(self, device):
        """Move layer to specified device"""
        super(KANLayer, self).to(device)
        self.device = device
        return self

    def forward(self, x):
        """
        Double-KAN Layer forward pass implementing Eq(11): spline_NB(x) = spline_b(x, W'_I) / spline_b(x, W'_K)
        
        Mathematical Implementation:
        - spline_b(x, W'_I) = sum_{i=0 to N} w'_i B_i(x)  (numerator)
        - spline_b(x, W'_K) = sum_{k=0 to N} w'_k B_k(x)  (denominator)
        - w'_k ≈ sqrt(w_k^2 + ζ) with ζ = 10^-8 for numerical stability
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (batch_size, input_dimension)
            
        Returns:
        --------
            y : 2D torch.float
                outputs, shape (batch_size, output_dimension)
        """
        
        # Base function computation
        base = self.base_fun(x)  # (batch_size, in_dim)
        
        # Compute spline_b(x, W'_I)
        spline_b_WI = coef2curve(x_eval=x, grid=self.grid, coef=self.WI, k=self.k)
        
        # Compute spline_b(x, W'_K)
        spline_b_WK = coef2curve(x_eval=x, grid=self.grid, coef=self.WK ** 2, k=self.k)
        spline_b_WK = F.softplus(spline_b_WK) + 1e-6  # Ensure positive with softplus
        
        spline_NB = spline_b_WI / (spline_b_WK + 1e-8)  # ζ = 10^-8 for numerical stability

        # Combine with base function and apply scaling
        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * spline_NB
        y = self.mask[None, :, :] * y

        # Sum over input dimension
        y = torch.sum(y, dim=1)
        return y



