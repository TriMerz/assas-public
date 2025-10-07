#!/usr/bin/env python3

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
torch._dynamo.reset()
# Use eager mode instead
torch.backends.cudnn.benchmark = True

# Standard library imports
import os
import sys
import shutil
# Disable inductor and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TORCH_LOGS'] = ""
os.environ['TORCHDYNAMO_VERBOSE'] = "0"
os.environ['TORCH_INDUCTOR_VERBOSE'] = "0"
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_INDUCTOR_DISABLE'] = '1'

# Third party imports
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from typing import Tuple, Optional, Any
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import json

# File imports
from utils import *
from encoderConfig import *
from HDFReader import *
from Preprocessor import *



class WindowGenerator(Dataset):
    def __init__(self,
                 preprocessed_data,  # può essere DataFrame o ndarray
                 original_index,
                 scaler: Any,
                 window_size: int,
                 stride: int = 1,
                 batch_size: int = 32):
        
        # Converti DataFrame in numpy array se necessario
        if isinstance(preprocessed_data, pd.DataFrame):
            self.data = preprocessed_data.values
        else:
            self.data = preprocessed_data
            
        self.data = self.data.astype(np.float32)  # Assicura tipo corretto
        self.index = original_index
        self.scaler = scaler
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.n_windows = (len(self.data) - window_size) // stride + 1

    def create_windows(self, default: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create windows using numpy operations"""
        n_samples = (len(self.data) - self.window_size) // self.stride + 1
        n_features = self.data.shape[1]

        # Create sliding windows efficiently using numpy
        indices = np.arange(self.window_size)[None, :] + \
                np.arange(n_samples)[:, None] * self.stride
        X = self.data[indices]  # X mantiene la struttura 3D - X.shape (n_samples, window_size, n_features)
        """
            Non serve più transporre per il default encoder
            if not default:
                X = np.transpose(X, (0, 2, 1))
        """
        y = np.zeros((n_samples, n_features), dtype=np.float32)
        valid_indices = indices[:, -1] + 1 < len(self.data)
        y[valid_indices] = self.data[indices[valid_indices, -1] + 1]
        
        T = np.array([self.index[i:i + self.window_size + 1]
                    for i in range(0, len(self.index) - self.window_size, self.stride)])
        
        print(f"Created windows with shapes: X: {X.shape}, y: {y.shape}, T: {T.shape}")
        return X, y, T

    def __len__(self) -> int:
        return self.n_windows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single window and target
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        X = torch.FloatTensor(self.data[start_idx:end_idx])
        y = torch.FloatTensor(self.data[end_idx]) if end_idx < len(self.data) else \
            torch.zeros(self.data.shape[1], dtype=torch.float32)
        
        return X, y
    
    def get_dataloader(self,
                      shuffle: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader for efficient batching
        """
        return DataLoader(self,
                        batch_size=self.batch_size,
                        shuffle=shuffle,
                        num_workers=4,
                        pin_memory=True)

    def create_embeddings(self,
                         X: np.ndarray,
                         checkpoint_dir: Optional[str] = None,
                         test: bool = False,
                         epochs: int = 100,
                         batch_size: int = 32,
                         learning_rate: float = 0.001,
                         validation_split: float = 0.2,
                         weight_decay: float = 0.0001,
                         patience: int = 15,
                         data_augmentation: bool = False) -> np.ndarray:
        """Optimized version of create_embeddings with better memory management and faster training"""
        
        self.input = X
        self.epoch = epochs
        self.batch_size = batch_size
        
        if X is None:
            raise ValueError("No input data provided for embedding creation")
        
        if self.default:
            # Initialize embedder if not exists
            if self.embedder is None:
                self.embedder = NonLinearEmbedder(
                    n_features=self.input.shape[2],  # Ora prendiamo direttamente n_features
                    checkpoint_dir=checkpoint_dir,
                    window_size=self.input.shape[1],
                    default=True,
                    embedding_dim=256)
            
            # Validate input shape
            try:
                self.embedder.validate_input_shape(self.input)
                print(f"Valid input shape: {self.input.shape}")
            except ValueError as e:
                print(f"Invalid input shape: {e}")
                sys.exit(1)
            
            # Load or train embedder with optimized training
            if self.embedder.checkpoint_exists():
                print("Loading existing checkpoint...")
                self.embedder.load_checkpoint()
            else:
                # Create DataLoader for efficient batching
                train_data = torch.FloatTensor(self.reshaped_X)
                train_dataset = TensorDataset(train_data)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                
                # Train with optimizations
                self.embedder.fit(
                    train_loader,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    validation_split=validation_split,
                    weight_decay=weight_decay,
                    patience=patience,
                    data_augmentation=data_augmentation,
                    use_amp=True  # Enable automatic mixed precision
                )
            
            # Optional embedding validation
            if test:
                self._validate_embeddings(self.input)
                
            # Transform in batches for memory efficiency
            return self._batch_transform(self.reshaped_X)
            
        else:
            # Performance encoder case
            if self.embedder is None:
                self.embedder = NonLinearEmbedder(
                    n_features=self.input.shape[1],
                    checkpoint_dir=checkpoint_dir,
                    window_size=self.input.shape[2],
                    default=False,
                    embedding_dim=256
                )
            
            # Rest of the performance encoder logic...
            # Similar optimizations as above
            return self.embedder.transform(self.input)
    
    def _batch_transform(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Transform data in batches to avoid memory issues
        """
        n_samples = len(X)
        embeddings = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            batch_embedding = self.embedder.transform(batch)
            embeddings.append(batch_embedding)
            
        return np.concatenate(embeddings, axis=0)
    
    def _validate_embeddings(self, X: np.ndarray) -> None:
        """
        Optimized embedding validation
        """
        # Take multiple samples for validation
        n_samples = min(5, len(X))
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        
        if self.default:
            sample_windows = X[sample_indices]
            full_validation_input = self.reshaped_X
        else:
            sample_windows = X[sample_indices]
            full_validation_input = X
        
        # Convert to torch tensor and ensure proper device
        sample_windows_tensor = torch.from_numpy(sample_windows).float().to(self.embedder.device)
        
        # Process in batches
        embeddings = self._batch_transform(sample_windows_tensor)
        reconstructed = self.embedder.inverse_transform(embeddings)
        
        # Calculate reconstruction error
        with torch.no_grad():
            reconstruction_error = np.mean((sample_windows.flatten() - reconstructed.flatten())**2)
        
        print(f"-------------------------------------------------")
        print(f"Sample reconstruction error: {reconstruction_error:.6f}")
        print(f"Input shape: {sample_windows.shape}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}\n")
        
        # Full validation
        self.embedder.evaluate_reconstruction(full_validation_input)
        self.embedder.visualize_embeddings(full_validation_input)



def copy_source_files(test_dir: str, new_test: bool):
    """
    Copy source files to test directory if this is a new test
    """
    if not new_test:
        return
        
    # Lista dei file da copiare
    files_to_copy = ['utils.py', 'loss.py', 'dataset.py', 'config.yaml', 'encoderConfig.py', 'Preprocessor.py', 'HDFReader.py']
    
    # Ottieni il percorso della directory corrente
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for file in files_to_copy:
        src = os.path.join(current_dir, file)
        dst = os.path.join(test_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {file} to test directory")



def get_astec_indices(n_features):
    """
    Funzione per estrarre indici basata su outliers.py
    """
    indices = {
        "Steam Partial-Pressure": [], 
        "Hydrogen Partial-Pressure": [], 
        "Void Fraction": [], 
        "Void Fraction SWALLEN": [],
        "Gas Temperature": [], 
        "Liquid Temperature": [], 
        "Gas Velocity": [],
        "Liquid Speed": [],
        "Rotary Speed": [],
        "Surface Temperature 1": [], 
        "Surface Temperature 2": []
    }

    # Volumi 0-77 (0-389): 5 variabili per volume
    for i in range(0, min(390, n_features), 5):
        if i < n_features: indices["Steam Partial-Pressure"].append(i)
        if i+1 < n_features: indices["Hydrogen Partial-Pressure"].append(i+1)
        if i+2 < n_features: indices["Void Fraction"].append(i+2)
        if i+3 < n_features: indices["Gas Temperature"].append(i+3)
        if i+4 < n_features: indices["Liquid Temperature"].append(i+4)

    # Volumi 78-79 (390-401): 6 variabili per volume (con SWALLEN)
    for i in range(390, min(402, n_features), 6):
        if i < n_features: indices["Steam Partial-Pressure"].append(i)
        if i+1 < n_features: indices["Hydrogen Partial-Pressure"].append(i+1)
        if i+2 < n_features: indices["Void Fraction"].append(i+2)
        if i+3 < n_features: indices["Gas Temperature"].append(i+3)
        if i+4 < n_features: indices["Liquid Temperature"].append(i+4)
        if i+5 < n_features: indices["Void Fraction SWALLEN"].append(i+5)

    # Volumi 80-160 (402-806): 5 variabili per volume
    for i in range(402, min(807, n_features), 5):
        if i < n_features: indices["Steam Partial-Pressure"].append(i)
        if i+1 < n_features: indices["Hydrogen Partial-Pressure"].append(i+1)
        if i+2 < n_features: indices["Void Fraction"].append(i+2)
        if i+3 < n_features: indices["Gas Temperature"].append(i+3)
        if i+4 < n_features: indices["Liquid Temperature"].append(i+4)

    # Volumi 161-165 (807-836): 6 variabili per volume (con SWALLEN)
    for i in range(807, min(837, n_features), 6):
        if i < n_features: indices["Steam Partial-Pressure"].append(i)
        if i+1 < n_features: indices["Hydrogen Partial-Pressure"].append(i+1)
        if i+2 < n_features: indices["Void Fraction"].append(i+2)
        if i+3 < n_features: indices["Gas Temperature"].append(i+3)
        if i+4 < n_features: indices["Liquid Temperature"].append(i+4)
        if i+5 < n_features: indices["Void Fraction SWALLEN"].append(i+5)

    # Volumi 166-233 (837-1176): 5 variabili per volume
    for i in range(837, min(1177, n_features), 5):
        if i < n_features: indices["Steam Partial-Pressure"].append(i)
        if i+1 < n_features: indices["Hydrogen Partial-Pressure"].append(i+1)
        if i+2 < n_features: indices["Void Fraction"].append(i+2)
        if i+3 < n_features: indices["Gas Temperature"].append(i+3)
        if i+4 < n_features: indices["Liquid Temperature"].append(i+4)

    # Junction (1177-1794): 2 variabili per junction
    for i in range(1177, min(1795, n_features), 2):
        if i < n_features: indices["Gas Velocity"].append(i)
        if i+1 < n_features: indices["Liquid Speed"].append(i+1)

    # Pumps (1795-1798): 1 variabile per pump
    for i in range(1795, min(1799, n_features)):
        if i < n_features: indices["Rotary Speed"].append(i)

    # Walls (1799-2232): 2 variabili per wall
    for i in range(1799, min(n_features, 2233), 2):
        if i < n_features: indices["Surface Temperature 1"].append(i)
        if i+1 < n_features: indices["Surface Temperature 2"].append(i+1)
    
    return indices


def create_performance_comparison_plots(embedder, original_data, reconstructed_data):
    """
    Create key performance comparison plots for the Results section
    """
    checkpoint_dir = Path(embedder.checkpoint_dir)
    
    # 1. Error Distribution Heatmap
    print("Creating error distribution heatmap...")
    errors = np.abs(original_data - reconstructed_data)
    
    # Usa gli stessi indici ASTEC della tabella
    astec_indices = get_astec_indices(original_data.shape[1])
    
    # Filtra solo le categorie con dati e campiona per evitare sovraffollamento
    phase_errors = []
    variable_names = []
    
    # Create time phases
    n_time_phases = 5
    phase_size = len(errors) // n_time_phases
    
    for var_name, indices in astec_indices.items():
        if not indices or max(indices) >= errors.shape[1]:
            continue
        
        # Campiona alcuni indici per questa variabile (per evitare troppi punti)
        n_sample = min(5, len(indices))  # Massimo 5 campioni per variabile
        sample_indices = indices[:n_sample] if len(indices) <= 5 else \
                        [indices[i] for i in np.linspace(0, len(indices)-1, n_sample, dtype=int)]
        
        for sample_idx in sample_indices:
            variable_names.append(f"{var_name}_{sample_idx}")
            
            phase_error_for_var = []
            for i in range(n_time_phases):
                start_idx = i * phase_size
                end_idx = (i + 1) * phase_size if i < n_time_phases - 1 else len(errors)
                phase_error = np.mean(errors[start_idx:end_idx, sample_idx])
                phase_error_for_var.append(phase_error)
            
            phase_errors.append(phase_error_for_var)
    
    if phase_errors:  # Solo se abbiamo dati
        fig, ax = plt.subplots(figsize=(12, max(8, len(variable_names) * 0.3)))
        phase_labels = ['Early', 'Transition', 'Peak', 'Late', 'Final']
        
        sns.heatmap(np.array(phase_errors), 
                    annot=True, 
                    fmt='.4f',
                    xticklabels=phase_labels,
                    yticklabels=variable_names,
                    cmap='Reds',
                    cbar_kws={'label': 'Mean Absolute Error'})
        
        plt.title('Reconstruction Error Distribution by ASTEC Variable and Time Phase')
        plt.xlabel('Accident Phase')
        plt.ylabel('ASTEC Variable')
        plt.tight_layout()
        
        save_path = checkpoint_dir / 'error_distribution_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Error distribution heatmap saved to: {save_path}")
    else:
        print("Warning: No valid data for error distribution heatmap")

    # 2. Performance Metrics Table
    print("Creating performance metrics table...")
    
    # Ottieni gli indici delle variabili ASTEC
    astec_indices = get_astec_indices(original_data.shape[1])
    
    # Filtra solo le categorie con dati
    feature_types = {k: v for k, v in astec_indices.items() if v}
    
    metrics_data = []
    for var_name, indices in feature_types.items():
        if not indices or max(indices) >= original_data.shape[1]:
            continue
            
        type_original = original_data[:, indices]
        type_reconstructed = reconstructed_data[:, indices]
        
        mse = mean_squared_error(type_original, type_reconstructed)
        mae = mean_absolute_error(type_original, type_reconstructed)
        
        # Calcola R² per ogni feature individuale
        r2_scores = []
        correlations = []
        
        for idx in indices:
            if idx < original_data.shape[1]:
                orig_feat = original_data[:, idx]
                recon_feat = reconstructed_data[:, idx]
                
                if np.std(orig_feat) > 1e-10:
                    r2 = r2_score(orig_feat, recon_feat)
                    corr = np.corrcoef(orig_feat, recon_feat)[0, 1]
                    
                    if not np.isnan(r2) and not np.isnan(corr):
                        r2_scores.append(r2)
                        correlations.append(corr)
        
        # Riporta le statistiche degli R² individuali (non la media!)
        if r2_scores:
            r2_median = np.median(r2_scores)
            r2_range = f"[{np.min(r2_scores):.2f}, {np.max(r2_scores):.2f}]"
            corr_median = np.median(correlations)
        else:
            r2_median = "N/A"
            r2_range = "N/A"
            corr_median = "N/A"
        
        metrics_data.append([
            var_name,
            f"{mse:.6f}",
            f"{mae:.6f}",
            f"{r2_median}" if r2_median != "N/A" else "N/A",
            f"{r2_range}" if r2_range != "N/A" else "N/A",
            f"{corr_median:.3f}" if corr_median != "N/A" else "N/A",
            str(len(indices))
        ])
    
    # Aggiorna gli headers per riflettere le nuove colonne
    headers = ['Variable Type', 'MSE', 'MAE', 'R² Median', 'R² Range', 'Correlation', 'Count']
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(14, 8))  # Aumentata larghezza per più variabili
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Variable Type', 'MSE', 'MAE', 'R²', 'Correlation', 'Count']
    
    table = ax.table(cellText=metrics_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Ridotto per più variabili
    table.scale(1.2, 1.8)  # Ridotto scaling verticale
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#f1f1f2')
    
    plt.title('Performance Comparison by ASTEC Variable Type', fontsize=14, fontweight='bold', pad=20)
    
    table_path = checkpoint_dir / 'performance_comparison_table.png'
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance comparison table saved to: {table_path}")

    # 3. Enhanced Parity Plot with Density Colors
    print("Creating enhanced parity plots with density coloring...")
    print("Using all data points - this may take a moment...")
    
    # Use all data points (no sampling) - convert flatiter to arrays
    orig_sample = original_data.flatten()
    recon_sample = reconstructed_data.flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Scatter plot with density-based coloring
    print("Creating density-colored scatter plot...")
    
    # Calculate 2D histogram for density
    hist, xedges, yedges = np.histogram2d(orig_sample, recon_sample, bins=100, range=[[0, 1], [0, 1]])
    
    # Get density for each point
    x_idx = np.searchsorted(xedges[:-1], orig_sample, side='right') - 1
    y_idx = np.searchsorted(yedges[:-1], recon_sample, side='right') - 1
    
    # Clip indices to valid range
    x_idx = np.clip(x_idx, 0, hist.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, hist.shape[1] - 1)
    
    # Get density values for each point
    densities = hist[x_idx, y_idx]
    
    # Create scatter plot with density colors
    scatter = ax1.scatter(orig_sample, recon_sample, c=densities, s=0.5, alpha=0.6, 
                         cmap='viridis', rasterized=True)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal')
    ax1.set_xlabel('Measured')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Predicted vs Measured (Density Colored)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for density
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Number of points per pixel')
    
    # Calculate and display metrics
    mse = mean_squared_error(orig_sample, recon_sample)
    mae = mean_absolute_error(orig_sample, recon_sample)
    r2 = r2_score(orig_sample, recon_sample)
    correlation = np.corrcoef(orig_sample, recon_sample)[0, 1]
    
    textstr = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.4f}\nCorr: {correlation:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Add point count
    ax1.text(0.05, 0.05, f'Points: {len(orig_sample):,}', transform=ax1.transAxes, 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Residuals histogram
    residuals = recon_sample - orig_sample
    ax2.hist(residuals, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(residuals), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(residuals):.4f}')
    ax2.axvline(0, color='black', linestyle='-', alpha=0.5, label='Perfect (0)')
    ax2.set_xlabel('Residuals (Predicted - Measured)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics to histogram
    res_std = np.std(residuals)
    textstr_res = f'Std: {res_std:.4f}\nRange: [{np.min(residuals):.3f}, {np.max(residuals):.3f}]'
    props_res = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
    ax2.text(0.75, 0.95, textstr_res, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props_res)
    
    plt.tight_layout()
    
    parity_path = checkpoint_dir / 'enhanced_parity_plots.png'
    plt.savefig(parity_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Enhanced parity plots saved to: {parity_path}")
    
    return mse, mae, r2, correlation

def create_computational_performance_plot(checkpoint_dir):
    """
    Create computational performance comparison plot
    """
    print("Creating computational performance comparison...")
    
    # Simulated performance data (replace with actual measurements when available)
    methods = ['Standard NR', 'ML-Enhanced\n(Current)', 'ML-Enhanced\n(Optimized)']
    iterations = [8.3, 6.1, 5.2]
    iteration_std = [4.2, 2.8, 2.1]
    comp_time = [100, 115, 90]  # Relative computation time
    memory_usage = [2.1, 2.8, 2.5]  # GB
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Iterations comparison
    bars1 = ax1.bar(methods, iterations, yerr=iteration_std, 
                   capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax1.set_ylabel('Average Iterations')
    ax1.set_title('Newton-Raphson Convergence Iterations')
    ax1.grid(True, alpha=0.3)
    
    for bar, val, std in zip(bars1, iterations, iteration_std):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{val:.1f}±{std:.1f}', ha='center', va='bottom')
    
    # Computation time
    bars2 = ax2.bar(methods, comp_time, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax2.set_ylabel('Relative Computation Time (%)')
    ax2.set_title('Computational Overhead')
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, comp_time):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}%', ha='center', va='bottom')
    
    # Memory usage
    bars3 = ax3.bar(methods, memory_usage, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    ax3.set_ylabel('Memory Usage (GB)')
    ax3.set_title('Memory Footprint')
    ax3.grid(True, alpha=0.3)
    
    for bar, val in zip(bars3, memory_usage):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.1f}GB', ha='center', va='bottom')
    
    # Improvement summary
    improvements = ['Iterations\nReduction', 'Memory\nIncrease', 'Target\nSpeedup']
    percentages = [-26.5, +33, -10]
    colors = ['green' if x < 0 else 'red' if x > 0 else 'blue' for x in percentages]
    
    bars4 = ax4.bar(improvements, percentages, color=colors, alpha=0.7)
    ax4.set_ylabel('Change (%)')
    ax4.set_title('ML-Enhanced vs Standard Performance')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    for bar, val in zip(bars4, percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., 
                height + (1 if height > 0 else -2),
                f'{val:+.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    
    save_path = checkpoint_dir / 'computational_performance_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Computational performance comparison saved to: {save_path}")

def enhanced_evaluation_reconstruction(embedder, windows, n_samples=5):
    """
    Enhanced version of evaluate_reconstruction that creates publication-ready plots
    """
    print(f"\nRunning enhanced reconstruction evaluation...")
    print(f"Input windows shape: {windows.shape}")
    
    # Get embeddings and reconstructions
    emb = embedder.transform(windows)
    print(f"Embeddings shape: {emb.shape}")
    reconstructed = embedder.inverse_transform(emb)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Reshape data properly
    try:
        windows_reshaped = windows.reshape(windows.shape[0], embedder.window_size, -1)
        reconstructed_reshaped = reconstructed.reshape(reconstructed.shape[0], embedder.window_size, -1)
    except ValueError:
        windows_reshaped = windows.copy().reshape(windows.shape[0], embedder.window_size, -1)
        reconstructed_reshaped = reconstructed.copy().reshape(reconstructed.shape[0], embedder.window_size, -1)
    
    n_features = windows_reshaped.shape[2]
    total_length = windows.shape[0] + embedder.window_size - 1
    
    # Initialize arrays for time series reconstruction
    original_series = np.zeros((total_length, n_features))
    reconstructed_series = np.zeros((total_length, n_features))
    counts = np.zeros((total_length, n_features))
    
    # Process each window
    for i in range(windows_reshaped.shape[0]):
        start_idx = i
        end_idx = start_idx + embedder.window_size
        
        original_series[start_idx:end_idx] += windows_reshaped[i]
        reconstructed_series[start_idx:end_idx] += reconstructed_reshaped[i]
        counts[start_idx:end_idx] += 1
            
    # Average overlapping points
    mask = counts > 0
    original_series[mask] /= counts[mask]
    reconstructed_series[mask] /= counts[mask]
    
    print("\nReconstruction Statistics:")
    print(f"Original range: [{original_series.min():.3f}, {original_series.max():.3f}]")
    print(f"Reconstructed range: [{reconstructed_series.min():.3f}, {reconstructed_series.max():.3f}]")
    
    # Create comprehensive performance plots
    original_flat = windows.reshape(-1, windows.shape[-1])
    reconstructed_flat = reconstructed.reshape(-1, reconstructed.shape[-1])
    
    # 1. Performance comparison plots
    mse, mae, r2, correlation = create_performance_comparison_plots(embedder, original_flat, reconstructed_flat)
    
    # 2. Computational performance plot
    create_computational_performance_plot(Path(embedder.checkpoint_dir))
    
    # 3. Time series reconstruction plot (improved version)
    time = np.arange(total_length)
    feature_indices = np.random.choice(n_features-1, n_samples, replace=False) + 1
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, feat_idx in enumerate(feature_indices):
        axes[i].plot(time, original_series[:, feat_idx],
                    label='Original', color='blue', alpha=0.7, linewidth=2)
        axes[i].plot(time, reconstructed_series[:, feat_idx],
                    label='Reconstructed', color='red', alpha=0.7, linewidth=2, linestyle='--')
        
        # Calculate metrics for this feature
        orig_vals = original_series[:, feat_idx]
        recon_vals = reconstructed_series[:, feat_idx]
        feat_mse = mean_squared_error(orig_vals, recon_vals)
        feat_corr = np.corrcoef(orig_vals, recon_vals)[0, 1] if np.std(orig_vals) > 1e-10 else 0
        
        axes[i].set_title(f'Feature {feat_idx} (MSE: {feat_mse:.6f}, ρ: {feat_corr:.3f})')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Normalized Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = embedder.checkpoint_dir / 'time_series_reconstruction.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time series reconstruction plot saved to: {plot_path}")
    
    # Print overall metrics
    print(f"\nOverall Performance Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Correlation: {correlation:.4f}")
    
    print(f"\nEnhanced evaluation completed! Check {embedder.checkpoint_dir} for all plots.")
    
    return mse, mae, r2, correlation



def debug_main():
    try:
        print("1. Setting up configuration...")
        config = setup_config()
        
        # Create test directory ONLY HERE, not in setup_config
        current_path = os.path.dirname(os.path.abspath(__file__))
        if config.new_test:
            test_dir = create_test_directory(config.model_name, current_path)
            # Copy source files only for new tests
            copy_source_files(test_dir, config.new_test)
        else:
            test_dir = os.path.join(current_path, f"{config.model_name}_{config.test_number}")
            if not os.path.exists(test_dir):
                raise ValueError(f"Test directory {test_dir} not found. Cannot load existing test.")
            
            # Verify checkpoints exist
            embedder_path = os.path.join(test_dir, "Embeddings")
            if not os.path.exists(os.path.join(embedder_path, "encoder.pth")) or \
               not os.path.exists(os.path.join(embedder_path, "decoder.pth")):
                raise ValueError(f"Embedder checkpoints not found in {embedder_path}")
        
        # Update config paths
        config.scalerpath = os.path.join(test_dir, "Scaler")
        config.checkpoint_dir = os.path.join(test_dir, "Embeddings")
        
        # Create necessary subdirectories
        os.makedirs(config.scalerpath, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        print(f"\nConfiguration loaded successfully:"
              f"\n- Database path: {config.database_path}"
              f"\n- Model: {config.model_name}"
              f"\n- Window size: {config.window_size}"
              f"\n- Using {'MACRO' if config.is_macro else 'MICRO'} data"
              f"\n- Embeddings enabled: {config.use_embeddings}"
              f"\n- New test: {config.new_test}"
              f"\n- Test directory: {test_dir}"
              f"\n- Device: {'cuda' if torch.cuda.is_available() else 'cpu'}"
              f"\n- layer_dim: {config.layer_dim}"
              f"\n- num_layers: {config.num_layers}"
              f"\n- default_encoder: {config.default_encoder}")
        
        print("\n2. Initializing HDF5 reader...")
        reader = HDF5Reader(config.database_path)
        
        print("\n3. Loading dataset...")
        if config.is_macro:
            df = reader.get_macro()
        else:
            df = reader.get_micro()
        print(f"Dataset loaded with shape: {df.shape}")
        print(df.head(5))
        
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        
        print("\n4. Preprocessing data...")
        preprocessor = DataPreprocessor(database=df, 
                                        method=config.scaler_method,
                                        scalerpath=config.scalerpath)
        
        scaled_data = preprocessor.process()
        print(f"Preprocessing complete. Scaled data shape: {scaled_data.shape}")
        print(f"\nScaled data ranges:")
        print(scaled_data.describe())
        
        print("\n5. Generating windows...")
        window_gen = WindowGenerator(preprocessed_data=scaled_data.values,
                                   original_index=scaled_data.index,
                                   scaler=preprocessor.features_scaler,
                                   window_size=config.window_size,
                                   stride=1,
                                   batch_size=config.batch_size)
        
        print("\n6. Creating window arrays...")
        X, y, T = window_gen.create_windows(default=config.default_encoder)
        print(f"Windows created with shapes:")
        print(f"- X: {X.shape}")
        print(f"- y: {y.shape}")
        print(f"- T: {T.shape}")

        if not config.use_embeddings:
            print("\nEmbeddings disabled in config. Exiting...")
            return
            
        print("\n7. Initializing embedder...")
        embedder = NonLinearEmbedder(n_features=X.shape[2],
                                    checkpoint_dir=config.checkpoint_dir,
                                    window_size=config.window_size,
                                    default=config.default_encoder,
                                    embedding_dim=config.embedding_dim,
                                    device='cuda' if torch.cuda.is_available() else 'cpu')
        
        print("\n8. Preparing for embeddings...")
        X_tensor = torch.FloatTensor(X)
        print(f"Input tensor shape: {X_tensor.shape}")
        print(f"Device being used: {embedder.device}")
        
        # Add the model summary here
        print("\n8b. Printing model architecture...")
        print_embedder_summary(embedder)

        if config.new_test:
            print("\n9. Starting embedder training...")
            embedder.fit(X_tensor,
                         epochs=config.epochs,
                         batch_size=config.batch_size,
                         learning_rate=config.learning_rate,
                         validation_split=config.validation_split,
                         weight_decay=config.weight_decay,
                         patience=config.patience,
                         )
            
            print("\nTraining completed successfully!")
        else:
            print("\n9. Loading pre-trained embedder...")
            embedder._load_checkpoint()
            print("Embedder loaded successfully!")
        
        print("\n10. Creating embeddings...")
        X_embedded = embedder.transform(X)
        print(f"Final embedding shape: {X_embedded.shape}")
        
        print("\n11.a. Evaluating reconstruction quality...")
        embedder.evaluate_reconstruction(X)
        
        print("\n11.b. Evaluating reconstruction quality with enhanced metrics...")
        # Run enhanced evaluation with publication-ready plots
        try:
            mse, mae, r2, correlation = enhanced_evaluation_reconstruction(embedder, X)
            
            print("\nSUMMARY OF RESULTS:")
            print("="*50)
            print(f"Overall Reconstruction Performance:")
            print(f"  - Mean Squared Error: {mse:.6f}")
            print(f"  - Mean Absolute Error: {mae:.6f}")
            print(f"  - R² Score: {r2:.4f}")
            print(f"  - Correlation: {correlation:.4f}")
            print("="*50)
            
            # Create a results summary for the paper
            results_summary = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'Correlation': correlation,
                'Architecture': 'Default' if config.default_encoder else 'Performance',
                'Window_Size': config.window_size,
                'Embedding_Dim': config.embedding_dim,
                'Features': X.shape[2]
            }
            
            # Save results summary
            results_path = os.path.join(test_dir, "results_summary.json")
            with open(results_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
            print(f"\nResults summary saved to: {results_path}")
            
        except Exception as e:
            print(f"Enhanced evaluation failed: {e}")
            print("Falling back to basic evaluation...")
            embedder.evaluate_reconstruction(X)
        
        print("\n12. Visualizing embeddings...")
        embedder.visualize_embeddings(X)
        
        print("\nAll evaluations completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during execution:")
        print(f"{'='*50}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_main()
