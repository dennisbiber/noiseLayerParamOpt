import torch
import torch.nn as nn
from torch.nn import Module
from typing import Tuple, Union
from torch import Tensor

class AddNoise(Module):
    def __init__(self, threshold: float, exponent: float, slope: float, intercept: float, 
                 noise_type: str = 'gaussian', mean: float = 0.0, std: float = 1.0,
                 grid_size: tuple = (1, 1), circle_size_factor: float = 1.0,
                 heightSkew: float = 1.0, widthSkew: float = 1.0, inversionBool: bool = True,
                 device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the AddNoise module.

        Args:
        - threshold (float): Threshold value above which noise is added.
        - exponent (float): Exponent for the noise generation formula.
        - slope (float): Slope coefficient for the noise generation formula.
        - intercept (float): Intercept coefficient for the noise generation formula.
        - noise_type (str): Type of noise to add. Currently supports 'gaussian'.
        - mean (float): Mean of the Gaussian noise.
        - std (float): Standard deviation of the Gaussian noise.
        """
        super().__init__()
        self.device = device
        self.threshold = threshold
        self.exponent = exponent
        self.slope = slope
        self.intercept = intercept
        self.noise_type = noise_type
        self.mean = mean
        self.std = std
        self.grid_size = grid_size
        self.image_size = None
        self.circle_size_factor = circle_size_factor
        self.hSkew = heightSkew
        self.wSkew = widthSkew
        self.invert = inversionBool

    def _generate_circle_masks(self) -> torch.Tensor:
        """
        Generate elliptical masks for a grid of circles.

        Returns:
        - Tensor: A tensor containing elliptical masks.
        """
        yH = self.grid_size[0]
        xW = self.grid_size[-1]
        circle_masks = torch.zeros(1, 1, *self.image_size, device=self.device)
        for i in range(yH):
            for j in range(xW):
                cy = (i + 0.5) * (self.image_size[0]/yH)
                cx = (j + 0.5) * (self.image_size[1]/xW)

                y_indices, x_indices = torch.meshgrid(torch.arange(self.image_size[0], device=self.device), torch.arange(self.image_size[1], device=self.device))
                y_distances = (y_indices - cy) / (self.image_size[0] / (2* yH) * self.circle_size_factor)*(1-self.hSkew)
                x_distances = (x_indices - cx) / (self.image_size[1] / (2* xW) * self.circle_size_factor)*(1-self.wSkew)
                distances = torch.sqrt(y_distances ** 2 + x_distances ** 2)
                circle_masks[0, 0, (distances <= 1)] = 1

        # Invert the mask
        if self.invert:
            return 1 - circle_masks
        return circle_masks

    def forward(self, x: Tensor) -> Tensor:
        """
        Add noise to the input tensor based on the threshold.

        Args:
        - x (Tensor): Input tensor (image).

        Returns:
        - Tensor: Noisy image.
        """
        # Calculate noise based on the threshold
        x = x.to(self.device)
        self.image_size = x.shape[-2:]
        above_threshold = (x > self.threshold).float()
        noise = (self.slope * x.pow(self.exponent) + self.intercept).pow(self.exponent) * above_threshold
        mask = self._generate_circle_masks()
        # Add Gaussian noise
        if self.noise_type == 'gaussian':
            noise_tensor = torch.randn_like(x) * self.std + self.mean
        elif self.noise_type == 'uniform':
            noise_tensor = torch.rand_like(x) * (self.std - self.mean) + self.mean
        elif self.noise_type == 'vonmises':
            u = torch.rand_like(x) * 2 * torch.pi
            noise_tensor = self.mean + torch.log(torch.exp(-self.std * torch.ones_like(x)) + (torch.exp(self.std * torch.ones_like(x)) - 1) * (1 - torch.cos(u))) / self.std        
        else:
            raise NotImplementedError(f"Noise type '{self.noise_type}' is not supported.")

        # Add noise to the input image
        noisy_image = x + (noise * noise_tensor) * mask

        return noisy_image