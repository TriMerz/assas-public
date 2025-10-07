#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss function for the autoencoder
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.1, gamma=0.1):
        """
        Modified loss with emphasis on pure reconstruction to enable overfitting testing
        
        Parameters:
        - alpha: weight for reconstruction loss (increased from 0.5 to 0.8)
        - beta: weight for feature correlation loss (reduced from 0.15 to 0.1)
        - gamma: weight for physics loss (reduced from 0.25 to 0.1)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        assert alpha + beta + gamma == 1.0, "Weights must sum to 1"
        
    def focal_mse(self, pred, target):
        """
        Simplified MSE without focal weighting to allow pure optimization
        """
        return F.mse_loss(pred, target)

    def temporal_consistency(self, reconstructed, original):
        """
        Relaxed temporal consistency to allow more flexibility in fitting
        """
        rec_diff = reconstructed[:, 1:] - reconstructed[:, :-1]
        orig_diff = original[:, 1:] - original[:, :-1]
        
        # Removed transition weighting to allow pure optimization
        return F.mse_loss(rec_diff, orig_diff)

    def feature_correlation_loss(self, reconstructed, original):
        """
        Simplified correlation loss with better numerical stability
        """
        # Add small epsilon to avoid numerical instability
        eps = 1e-8
        
        # Calculate correlations with stability measures
        rec_corr = self.batch_correlations(reconstructed + eps)
        orig_corr = self.batch_correlations(original + eps)
        
        return F.mse_loss(rec_corr, orig_corr)
    
    def physics_consistency(self, reconstructed, original):
        """
        Relaxed physics constraints to allow testing of pure fitting capacity
        """
        # Only preserve basic non-negativity where needed
        positive_mask = (original > 0)
        positivity_error = F.relu(-reconstructed[positive_mask]).mean()
        
        return positivity_error

    @staticmethod
    def batch_correlations(x):
        """
        Numerically stable correlation computation
        """
        # Add small epsilon for numerical stability
        eps = 1e-8
        
        # x shape: (batch, time, features)
        b, t, f = x.shape
        
        # Reshape and normalize
        x_flat = x.reshape(-1, f)
        mean = x_flat.mean(dim=0, keepdim=True)
        std = x_flat.std(dim=0, keepdim=True) + eps
        
        # Compute correlations with stability measures
        centered = (x_flat - mean) / std
        corr = (centered.T @ centered) / (b * t - 1)
        
        # Ensure correlations are in [-1, 1] range
        return torch.clamp(corr, -1 + eps, 1 - eps)

    def forward(self, reconstructed, original):
        """
        Modified forward pass focusing on reconstruction quality
        """
        # Normalize inputs for stability
        rec_norm = (reconstructed - reconstructed.mean()) / (reconstructed.std() + 1e-8)
        orig_norm = (original - original.mean()) / (original.std() + 1e-8)
        
        # Base reconstruction loss with pure MSE
        mse_loss = self.focal_mse(rec_norm, orig_norm)
        
        # Relaxed temporal consistency
        temp_loss = self.temporal_consistency(rec_norm, orig_norm)
        
        # Simplified feature correlations
        feat_loss = self.feature_correlation_loss(rec_norm, orig_norm)
        
        # Basic physics constraints
        physics_loss = self.physics_consistency(reconstructed, original)
        
        # Combined loss with emphasis on reconstruction
        total_loss = (self.alpha * mse_loss + 
                     self.beta * feat_loss +
                     self.gamma * physics_loss)
        
        # Add debug information during training
        if not self.training:
            print(f"\nLoss Components:")
            print(f"MSE Loss: {mse_loss.item():.6f}")
            print(f"Temporal Loss: {temp_loss.item():.6f}")
            print(f"Feature Correlation Loss: {feat_loss.item():.6f}")
            print(f"Physics Loss: {physics_loss.item():.6f}")
            print(f"Total Loss: {total_loss.item():.6f}")
            
        return total_loss
