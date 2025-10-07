#!/usr/bin/env python3

# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss import *
from pathlib import Path

from encoderConfig import *
from Preprocessor import *
from loss import *

config = setup_config()



class Default(nn.Module):
    def __init__(self, input_size, embedding_dim, num_layers=3, layer_dim=1024):
        super().__init__()
        self.window_size, self.n_features = input_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = layer_dim

        self.encoder = nn.Sequential(
            nn.LayerNorm(self.n_features),
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LSTM(input_size=self.hidden_dim,
                   hidden_size=self.hidden_dim // 2,  # Diviso 2 perché bidirezionale
                   num_layers=self.num_layers,
                   bidirectional=True,
                   batch_first=True,
                   dropout=0.1),
            # proiezione lineare per l'embedding
            nn.Linear(self.hidden_dim, self.embedding_dim),  # da hidden_dim a embedding_dim
            nn.LayerNorm(self.embedding_dim),  # Normalizzazione della dimensione corretta
            nn.Linear(self.embedding_dim, self.embedding_dim),  # mantiene la dimensione embedding
            nn.LayerNorm(self.embedding_dim)  # Normalizzazione finale
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LSTM(input_size=self.hidden_dim,
                   hidden_size=self.hidden_dim,
                   num_layers=self.num_layers,
                   batch_first=True,
                   dropout=0.1),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.n_features),
            nn.LayerNorm(self.n_features)
        )

    def encode(self, x):
        # x shape: (batch, time, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Forward through encoder layers sequentially
        x = self.encoder[0](x)  # LayerNorm
        x = self.encoder[1](x)  # Linear -> hidden_dim
        x = self.encoder[2](x)  # LayerNorm
        x = self.encoder[3](x)  # GELU
        x = self.encoder[4](x)  # Dropout
        x, (hidden, _) = self.encoder[5](x)  # LSTM
        # Concatenate final hidden states
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)  # -> hidden_dim
        x = self.encoder[6](hidden_cat)  # Linear -> embedding_dim
        x = self.encoder[7](x)  # LayerNorm -> embedding_dim
        x = self.encoder[8](x)  # Linear -> embedding_dim
        embedding = self.encoder[9](x)  # LayerNorm -> embedding_dim
        
        return embedding

    def decode(self, embedding):
        # embedding shape: (batch, embedding_dim)
        x = self.decoder[0](embedding)  # Linear -> hidden_dim
        x = self.decoder[1](x)  # LayerNorm
        x = self.decoder[2](x)  # GELU
        x = self.decoder[3](x)  # Dropout
        
        # Expand for sequence length
        x = x.unsqueeze(1).repeat(1, self.window_size, 1)
        
        x, _ = self.decoder[4](x)  # LSTM
        x = self.decoder[5](x)  # LayerNorm
        x = self.decoder[6](x)  # Linear -> n_features
        reconstruction = self.decoder[7](x)  # LayerNorm
        
        return reconstruction

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        
        return embedding, reconstruction



class ProbAttention(nn.Module):
    def __init__(self,
                 mask_flag=True,
                 factor=5,
                 scale=None,
                 attention_dropout=0.1,
                 output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q [B, H, L, D]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1])
        else:
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # Apply attention mask if provided
        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            scores_top = scores_top.masked_fill(
                attn_mask == 0,
                float('-inf')
            )

        # Apply softmax and dropout to attention scores
        scores_top = self.dropout(F.softmax(scores_top, dim=-1))

        # Get the weighted sum of values
        context = torch.matmul(scores_top, values)
        
        # Restore original shape
        context = context.transpose(2, 1)

        return context

class AttentionLayer(nn.Module):
    def __init__(self,
                 attention,
                 d_model,
                 n_heads,
                 d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        
        out = out.contiguous().view(B, L, -1)
        return self.out_projection(out)

class Performance(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_dim,
                 device= 'cuda' if torch.cuda.is_available() else 'cpu', 
                 n_heads=8,
                 num_encoder_layers=3,
                 dropout=0.1):
        super(Performance, self).__init__()
        self.window_size, self.n_features = input_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Enhanced temporal embedding to handle more time features
        self.temporal_embedding = nn.Sequential(
            nn.Linear(15, embedding_dim // 2),  # 15 = enhanced time features
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )
        self.feature_embedding = nn.Linear(self.n_features - 15, embedding_dim * 3 // 4)
        
        # Position encoding
        self.pos_encoder = nn.Parameter(
            torch.randn(1, self.window_size, embedding_dim)
        )
        
        # Initial normalization
        self.input_norm = nn.LayerNorm(self.n_features)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': AttentionLayer(
                    ProbAttention(True, 5, attention_dropout=dropout),
                    embedding_dim, n_heads),
                'conv1': nn.Conv1d(
                    embedding_dim, embedding_dim * 2, 
                    kernel_size=3, padding=1),
                'conv2': nn.Conv1d(
                    embedding_dim * 2, embedding_dim,
                    kernel_size=3, padding=1),
                'norm1': nn.LayerNorm(embedding_dim),
                'norm2': nn.LayerNorm(embedding_dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_encoder_layers)
        ])
        
        # Decoder for reconstruction
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': AttentionLayer(
                    ProbAttention(True, 5, attention_dropout=dropout),
                    embedding_dim, n_heads),
                'conv1': nn.Conv1d(
                    embedding_dim, embedding_dim * 2,
                    kernel_size=3, padding=1),
                'conv2': nn.Conv1d(
                    embedding_dim * 2, embedding_dim,
                    kernel_size=3, padding=1),
                'norm1': nn.LayerNorm(embedding_dim),
                'norm2': nn.LayerNorm(embedding_dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_encoder_layers)
        ])
        
        # Output projections
        self.temporal_output = nn.Linear(embedding_dim // 4, 2)
        self.feature_output = nn.Linear(embedding_dim * 3 // 4, self.n_features - 2)
        
        # Final normalization
        self.output_norm = nn.LayerNorm(self.n_features)

    def encode(self, x):
        # x shape: [batch_size, window_size, n_features]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Initial normalization
        x = self.input_norm(x)
        
        # Split temporal and feature data
        temporal_data = x[:, :, :15]
        feature_data = x[:, :, 15:]
        
        # Embed temporal and feature data separately
        temporal_embedded = self.temporal_embedding(temporal_data)
        feature_embedded = self.feature_embedding(feature_data)
        
        # Combine embeddings
        x = torch.cat([temporal_embedded, feature_embedded], dim=-1)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Process through encoder layers
        for layer in self.encoder_layers:
            # Self-attention
            attn_out = layer['attention'](x, x, x, None)
            x = layer['norm1'](x + layer['dropout'](attn_out))
            
            # Convolutional block
            conv_out = layer['conv1'](x.transpose(-1, -2))
            conv_out = F.gelu(conv_out)
            conv_out = layer['conv2'](conv_out).transpose(-1, -2)
            
            x = layer['norm2'](x + layer['dropout'](conv_out))
        
        # Global pooling for final embedding
        return x.mean(dim=1)

    def decode(self, embedding):
        # Expand embedding for sequence generation
        x = embedding.unsqueeze(1).repeat(1, self.window_size, 1)
        
        # Process through decoder layers
        for layer in self.decoder_layers:
            # Self-attention
            attn_out = layer['attention'](x, x, x, None)
            x = layer['norm1'](x + layer['dropout'](attn_out))
            
            # Convolutional block
            conv_out = layer['conv1'](x.transpose(-1, -2))
            conv_out = F.gelu(conv_out)
            conv_out = layer['conv2'](conv_out).transpose(-1, -2)
            
            x = layer['norm2'](x + layer['dropout'](conv_out))
        
        # Split embedding back into temporal and feature components
        temporal_embedded = x[:, :, :self.embedding_dim // 4]
        feature_embedded = x[:, :, self.embedding_dim // 4:]
        
        # Decode temporal and feature data separately
        temporal_out = self.temporal_output(temporal_embedded)
        feature_out = self.feature_output(feature_embedded)
        
        # Combine outputs
        x = torch.cat([temporal_out, feature_out], dim=-1)
        
        # Final normalization
        return self.output_norm(x)

    def forward(self, x):
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return embedding, reconstruction



class NonLinearEmbedder:
    """Optimized version of NonLinearEmbedder with better performance and memory management"""
    def __init__(self,
                n_features: int,
                checkpoint_dir: str,
                window_size: int,
                default: bool,
                embedding_dim: int = 256,
                device: str = None):
        """
        Initialize embedder with optimized settings
        """
        self.n_features = n_features
        self.window_size = window_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.default = default
        self.embedding_dim = embedding_dim
        
        # Optimize device selection
        if device is None:
            if torch.cuda.is_available():
                # Get the GPU with most free memory
                gpu_id = torch.cuda.current_device()
                torch.cuda.set_device(gpu_id)
                self.device = f'cuda:{gpu_id}'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        # Initialize encoder/decoder
        self.encoder = None
        self.decoder = None
        self.encoder_path = self.checkpoint_dir / "encoder.pth"
        self.decoder_path = self.checkpoint_dir / "decoder.pth"
        
        # Initialize encoder based on selected architecture
        input_dims = (self.window_size, self.n_features)  # Create input dimensions tuple
        
        if self.default:
            self._default_encoder(input_dims)
        else:
            self._performance_encoder(input_dims)
            
        # Move models to device
        self.to_device()
        
        # Initialize training components
        self.history = {'train_loss': [], 'val_loss': []}
        self.scaler = GradScaler(enabled=(self.device.startswith('cuda')))
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def to_device(self):
        """Move models to device efficiently"""
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        if self.decoder:
            self.decoder = self.decoder.to(self.device)
    
    #@torch.cuda.amp.autocast('cuda')
    def fit(self,
            windows,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 1e-4,
            validation_split: float = 0.2,
            weight_decay: float = 1e-4,
            patience: int = 20,
            use_amp: bool = True):
        """Training function with proper history logging"""
        self.validate_input_shape(windows)
        
        # Convert input to tensor if needed
        if not isinstance(windows, torch.Tensor):
            windows = torch.FloatTensor(windows)

        # Ensure 3D shape
        if len(windows.shape) == 2:
            windows = windows.unsqueeze(0)
        
        # Prepare data loaders
        train_data, val_data = self._prepare_data(windows, validation_split)
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                      lr=learning_rate,
                                      weight_decay=weight_decay
                                    )
        
        # Scheduler initialization
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=learning_rate,
                                                        epochs=epochs,
                                                        steps_per_epoch=len(train_loader)
                                                    )
        
        criterion = CombinedLoss(alpha=0.8,    # Bilanciamento tra ricostruzione e temporale
                                 beta=0.1,    # Peso per correlazioni tra features
                                 gamma=0.1     # Enfasi su eventi rari
                            ).to(self.device)
        
        best_val_loss = float('inf')
        patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': []}  # Reset history at start of training
        
        print(f"\nStarting training for {epochs} epochs:")
        print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Best':>6}")
        print("-" * 40)
        
        # Training loop
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader,
                                           optimizer,
                                           criterion,
                                           scheduler,
                                           use_amp
                                        )
            
            val_loss = self._validate_epoch(val_loader, criterion)
            
            # Log losses
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            is_best = val_loss < best_val_loss
            best_marker = "*" if is_best else ""
            
            print(f"{epoch+1:5d} {train_loss:12.6f} {val_loss:12.6f} {best_marker:>6}")
            
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                self._load_checkpoint()
                break
        
        # Plot training history
        self.plot_training_history()

    def _train_epoch(self, train_loader, optimizer, criterion, scheduler, use_amp):
        """Enhanced training with debugging info"""
        self.encoder.train()
        self.decoder.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                if use_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        embedding = self.encoder.encode(batch)
                        reconstruction = self.decoder.decode(embedding)
                        loss = criterion(reconstruction, batch)
                    
                    # Print debug info for first batch
                    if batch_idx == 0:
                        print(f"\nDebug info for first batch:")
                        print(f"Input range: [{batch.min():.3f}, {batch.max():.3f}]")
                        print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
                        print(f"Reconstruction range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")
                        print(f"Loss: {loss.item():.6f}")
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Forward pass
                    embedding = self.encoder.encode(batch)
                    reconstruction = self.decoder.decode(embedding)
                    loss = criterion(reconstruction, batch)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                total_loss += loss.item()
                        
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx + 1}:")
                print(f"Input shape: {batch.shape}")
                print(f"Device: {self.device}")
                print(f"Error: {str(e)}")
                raise e

        return total_loss / num_batches

    @torch.no_grad()
    def _validate_epoch(self, val_loader, criterion):
        """Optimized validation epoch"""
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device)
                
                # Forward pass
                embedding = self.encoder.encode(batch)
                reconstruction = self.decoder.decode(embedding)
                loss = criterion(reconstruction, batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _prepare_data(self, windows, validation_split):
        """Prepare data for training efficiently"""
        n_val = int(len(windows) * validation_split)
        indices = torch.randperm(len(windows))
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        if isinstance(windows, torch.Tensor):
            train_data = windows[train_indices]
            val_data = windows[val_indices]
        else:
            train_data = torch.FloatTensor(windows[train_indices])
            val_data = torch.FloatTensor(windows[val_indices])
        
        return train_data, val_data
    
    def _create_dataloader(self, data, batch_size, shuffle):
        """Create optimized DataLoader"""
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    
    def _save_checkpoint(self):
        """Save checkpoint with both encoder and decoder states"""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # Save encoder
        encoder_path = os.path.join(self.checkpoint_dir, "encoder.pth")
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'history': self.history
        }, encoder_path)
        
        # Save decoder
        decoder_path = os.path.join(self.checkpoint_dir, "decoder.pth")
        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'history': self.history
        }, decoder_path)
        
        print(f"Saved encoder to {encoder_path}")
        print(f"Saved decoder to {decoder_path}")

    def _load_checkpoint(self):
        """Load both encoder and decoder checkpoints"""
        encoder_path = os.path.join(self.checkpoint_dir, "encoder.pth")
        decoder_path = os.path.join(self.checkpoint_dir, "decoder.pth")
        
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            # Load encoder
            encoder_checkpoint = torch.load(encoder_path, map_location=self.device)
            self.encoder.load_state_dict(encoder_checkpoint['encoder_state_dict'])
            self.history = encoder_checkpoint['history']
            
            # Load decoder
            decoder_checkpoint = torch.load(decoder_path, map_location=self.device)
            self.decoder.load_state_dict(decoder_checkpoint['decoder_state_dict'])
            
            print("Loaded checkpoints successfully")
        else:
            raise FileNotFoundError(f"Checkpoint files not found in {self.checkpoint_dir}")

    def _default_encoder(self, input_dims):
        """
        Initialize default encoder architecture
        """
        self.encoder = Default(input_size=input_dims,
                               embedding_dim=self.embedding_dim,
                               num_layers=config.num_layers,  # Usa il parametro dalla config
                               layer_dim=config.layer_dim     # Usa il parametro dalla config
                            ).to(self.device)
        
        self.decoder = Default(input_size=input_dims,
                               embedding_dim=self.embedding_dim,
                               num_layers=config.num_layers,  # Usa il parametro dalla config
                               layer_dim=config.layer_dim     # Usa il parametro dalla config
                            ).to(self.device)
            
        # Optimize memory usage
        torch.cuda.empty_cache()

    def _performance_encoder(self, input_dims):
        """
        Initialize performance encoder architecture
        """

        self.encoder = Performance(input_size=input_dims,
                                   embedding_dim=self.embedding_dim,
                                   device=self.device
                                   ).to(self.device)
        
        self.decoder = Performance(input_size=input_dims,
                                   embedding_dim=self.embedding_dim,
                                   device=self.device
                                   ).to(self.device)
        
        # Optimize memory usage
        torch.cuda.empty_cache()

    def visualize_embeddings(self, windows, labels=None):
        """
        Embedding visualization using t-SNE
        """
        # Get embeddings efficiently using batch processing
        embeddings = self.transform(windows)
        
        # Reduce dimensionality for visualization
        n_samples = embeddings.shape[0]
        perplexity = min(30, n_samples - 1)
        
        # CPU version of t-SNE
        tsne = TSNE(n_components=2,
                    random_state=42,
                    perplexity=perplexity,
                    n_jobs=-1)  # Use all available CPU cores
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        if labels is not None:
            scatter = plt.scatter(embeddings_2d[:, 0],
                                  embeddings_2d[:, 1],
                                  c=labels,
                                  cmap='viridis',
                                  alpha=0.6)
            plt.colorbar(scatter)
        else:
            plt.scatter(embeddings_2d[:, 0],
                    embeddings_2d[:, 1],
                    alpha=0.6)
        
        plt.title('t-SNE visualization of embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # Save plot
        plot_path = self.checkpoint_dir / 'tsne_embeddings.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE Plot saved at: {plot_path}")

    def plot_training_history(self):
        """Enhanced training history visualization"""
        if not hasattr(self, 'history') or not self.history:
            print("No training history available to plot")
            return
            
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot losses
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        # Add moving average for smoothing
        window = min(5, len(epochs) // 10)  # Dynamic window size
        if window > 1:
            train_ma = np.convolve(self.history['train_loss'], 
                                np.ones(window)/window, 
                                mode='valid')
            val_ma = np.convolve(self.history['val_loss'], 
                                np.ones(window)/window, 
                                mode='valid')
            ma_epochs = epochs[window-1:]
            plt.plot(ma_epochs, train_ma, 'b--', alpha=0.5, label='Train MA')
            plt.plot(ma_epochs, val_ma, 'r--', alpha=0.5, label='Val MA')
        
        # Enhance plot
        plt.grid(True, alpha=0.3)
        plt.title('Training and Validation Loss Over Time', pad=20)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Use log scale if loss values vary significantly
        if max(self.history['train_loss']) / min(self.history['train_loss']) > 100:
            plt.yscale('log')
        
        # Add min/max annotations
        min_train = min(self.history['train_loss'])
        min_val = min(self.history['val_loss'])
        plt.annotate(f'Min Train: {min_train:.6f}', 
                    xy=(epochs[self.history['train_loss'].index(min_train)], min_train),
                    xytext=(10, 10), textcoords='offset points')
        plt.annotate(f'Min Val: {min_val:.6f}',
                    xy=(epochs[self.history['val_loss'].index(min_val)], min_val),
                    xytext=(10, -10), textcoords='offset points')
        
        # Save plot
        plot_path = self.checkpoint_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nTraining history plot saved to {plot_path}")

    def checkpoint_exists(self):
        """Check if checkpoint exists with proper error handling"""
        try:
            return self.encoder_path.exists() and self.decoder_path.exists()
        except Exception as e:
            print(f"Error checking checkpoint existence: {e}")
            return False

    def evaluate_reconstruction(self, windows, n_samples=5):
        """
        Reconstruction evaluation with equal weights for all points
        """
        print(f"\nInput windows shape: {windows.shape}")
        emb = self.transform(windows)
        print(f"Embeddings shape: {emb.shape}")
        reconstructed = self.inverse_transform(emb)
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        try:
            windows_reshaped = windows.reshape(windows.shape[0], self.window_size, -1)
            reconstructed_reshaped = reconstructed.reshape(reconstructed.shape[0], self.window_size, -1)
        except ValueError:
            windows_reshaped = windows.copy().reshape(windows.shape[0], self.window_size, -1)
            reconstructed_reshaped = reconstructed.copy().reshape(reconstructed.shape[0], self.window_size, -1)
        
        n_features = windows_reshaped.shape[2]
        total_length = windows.shape[0] + self.window_size - 1
        
        # Initialize arrays
        original_series = np.zeros((total_length, n_features))
        reconstructed_series = np.zeros((total_length, n_features))
        counts = np.zeros((total_length, n_features))
        
        # Process each window with equal weights
        for i in range(windows_reshaped.shape[0]):
            start_idx = i
            end_idx = start_idx + self.window_size
            
            # Add values directly without weights
            original_series[start_idx:end_idx] += windows_reshaped[i]
            reconstructed_series[start_idx:end_idx] += reconstructed_reshaped[i]
            counts[start_idx:end_idx] += 1
                
        # Average overlapping points
        mask = counts > 0
        original_series[mask] /= counts[mask]
        reconstructed_series[mask] /= counts[mask]
        
        # Print reconstruction stats
        print("\nReconstruction Statistics:")
        print(f"Original range: [{original_series.min():.3f}, {original_series.max():.3f}]")
        print(f"Reconstructed range: [{reconstructed_series.min():.3f}, {reconstructed_series.max():.3f}]")
        
        # Plotting
        time = np.arange(total_length)
        feature_indices = np.random.choice(n_features-1, n_samples, replace=False) + 1
        
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 4*n_samples))
        if n_samples == 1:
            axes = [axes]
        
        metrics_dict = {}
        for i, feat_idx in enumerate(feature_indices):
            # Plot
            axes[i].plot(time, original_series[:, feat_idx],
                        label='Original', color='blue', alpha=0.7, linewidth=2)
            axes[i].plot(time, reconstructed_series[:, feat_idx],
                        label='Reconstructed', color='red', alpha=0.7, linewidth=2, linestyle='--')
            
            axes[i].set_title(f'Feature {feat_idx}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Calculate metrics
            metrics_dict[feat_idx] = {
                'MSE': mean_squared_error(original_series[:, feat_idx], 
                                        reconstructed_series[:, feat_idx]),
                'MAE': mean_absolute_error(original_series[:, feat_idx], 
                                        reconstructed_series[:, feat_idx]),
                'Correlation': np.corrcoef(original_series[:, feat_idx],
                                        reconstructed_series[:, feat_idx])[0,1],
                'R2': r2_score(original_series[:, feat_idx],
                            reconstructed_series[:, feat_idx])
            }
        
        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'reconstruction_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Print metrics
        print("\nReconstruction Metrics:")
        for feat_idx, metric_values in metrics_dict.items():
            print(f"\nFeature {feat_idx}:")
            for metric_name, value in metric_values.items():
                print(f"  {metric_name}: {value:.6f}")

    def transform(self, windows):
        """Transform data to embeddings"""
        self.encoder.eval()
        
        # Convert to tensor if needed
        if isinstance(windows, np.ndarray):
            windows = torch.FloatTensor(windows)
        
        # Add batch dimension if needed
        if len(windows.shape) == 2:
            windows = windows.unsqueeze(0)
        
        # Process in batches
        batch_size = 1024
        embeddings = []
        
        with torch.no_grad():
            try:
                for i in range(0, len(windows), batch_size):
                    batch = windows[i:i + batch_size].to(self.device)
                    embedding = self.encoder.encode(batch)
                    embeddings.append(embedding.cpu())
                    
                final_embeddings = torch.cat(embeddings, dim=0)
                return final_embeddings.numpy()
                
            except RuntimeError as e:
                print(f"\nError during transform:")
                print(f"Input shape: {windows.shape}")
                print(f"Device: {self.device}")
                print(f"Error: {str(e)}")
                raise

    def inverse_transform(self, embeddings):
        """Reconstruct from embeddings"""
        self.decoder.eval()
        
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.FloatTensor(embeddings)
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings.unsqueeze(0)
        
        batch_size = 1024
        reconstructions = []
        
        with torch.no_grad():
            try:
                for i in range(0, len(embeddings), batch_size):
                    batch = embeddings[i:i + batch_size].to(self.device)
                    reconstruction = self.decoder.decode(batch)
                    reconstructions.append(reconstruction.cpu())
                    
                final_reconstructions = torch.cat(reconstructions, dim=0)
                return final_reconstructions.numpy()
                
            except RuntimeError as e:
                print(f"\nError during inverse transform:")
                print(f"Input shape: {embeddings.shape}")
                print(f"Device: {self.device}")
                print(f"Error: {str(e)}")
                raise

    def validate_input_shape(self, windows):
        """Validate input shape with improved error messages"""
        try:
            if not isinstance(windows, (np.ndarray, torch.Tensor)):
                raise ValueError("Input must be numpy array or torch tensor")
            
            # Convert to numpy for consistent handling
            if isinstance(windows, torch.Tensor):
                windows = windows.cpu().numpy()
            
            if len(windows.shape) not in [2, 3]:
                raise ValueError(f"Input must be 2D or 3D, got shape {windows.shape}")
            
            if len(windows.shape) == 2:
                # Single sample case
                if windows.shape != (self.window_size, self.n_features):
                    raise ValueError(
                        f"Expected shape (window_size, n_features) = "
                        f"({self.window_size}, {self.n_features}), "
                        f"got {windows.shape}"
                    )
            else:  # 3D case
                if windows.shape[1:] != (self.window_size, self.n_features):
                    raise ValueError(
                        f"Expected shape (batch, window_size, n_features) = "
                        f"(*, {self.window_size}, {self.n_features}), "
                        f"got {windows.shape}"
                    )
                    
            # Check for NaN values
            if np.isnan(windows).any():
                raise ValueError("Input contains NaN values")
            
            return True
            
        except Exception as e:
            print(f"\nError during shape validation:")
            print(f"Input shape: {windows.shape if hasattr(windows, 'shape') else 'unknown'}")
            print(f"Error: {str(e)}")
            raise

    def _batch_transform(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Transform data in batches to avoid memory issues"""
        n_samples = len(X)
        embeddings = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            if isinstance(batch, np.ndarray):
                batch = torch.FloatTensor(batch)
            batch = batch.to(self.device)
            
            with torch.no_grad():
                if self.default:
                    embedding = self.encoder.encode(batch)
                else:
                    embedding, _ = self.encoder.encode(batch)
                    
            embeddings.append(embedding.cpu().numpy())
            
        return np.concatenate(embeddings, axis=0)



def model_summary(model, input_size=None):
    """
    Print a summary of the model architecture similar to Keras' model.summary()
    
    Args:
        model: PyTorch model
        input_size: Tuple of input dimensions (excluding batch size) if needed for forward pass
    """
    def get_layer_info(layer):
        # Get number of trainable parameters
        num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        
        # Get output shape
        output_size = "?"  # Default if we can't determine
        
        return {
            'class_name': layer.__class__.__name__,
            'num_params': num_params,
            'output_shape': output_size
        }
    
    print("\nModel Architecture Summary")
    print("=" * 100)
    print(f"{'Layer (type)':<40} {'Output Shape':<20} {'Param #':<15} {'Connected to':<20}")
    print("=" * 100)
    
    total_params = 0
    trainable_params = 0
    
    # Iterate through named modules to get layer info
    for name, layer in model.named_modules():
        # Skip the root module and container modules
        if name == "" or isinstance(layer, (nn.Sequential, Default, Performance)):
            continue
            
        layer_info = get_layer_info(layer)
        
        # Calculate parameters
        params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        trainable_params += params
        total_params += sum(p.numel() for p in layer.parameters())
        
        # Format the layer name and type
        layer_name = f"{name} ({layer_info['class_name']})"
        
        # Get input connections
        connected_to = ""
        try:
            for param in layer.parameters():
                if hasattr(param, '_backward_hooks'):
                    connected_to = str([hook for hook in param._backward_hooks.keys()])
        except:
            pass
        
        # Print layer information
        print(f"{layer_name:<40} {layer_info['output_shape']:<20} {params:<15,d} {connected_to:<20}")
    
    print("=" * 100)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    
    # Calculate model size
    model_size = total_params * 4 / (1024 * 1024)  # Size in MB (assuming float32)
    print(f"Model size (MB): {model_size:.2f}")
    
    if hasattr(model, 'device'):
        print(f"Model device: {model.device}")
    else:
        print(f"Model device: {next(model.parameters()).device}")

def print_embedder_summary(embedder):
    """
    Print summary for both encoder and decoder of the embedder
    
    Args:
        embedder: NonLinearEmbedder instance
    """
    print("\n" + "="*40 + " ENCODER " + "="*40)
    model_summary(embedder.encoder)
    
    print("\n" + "="*40 + " DECODER " + "="*40)
    model_summary(embedder.decoder)
    
    # Print additional embedder information
    print("\nEmbedder Configuration:")
    print(f"Architecture: {'Default' if embedder.default else 'Performance'}")
    print(f"Window Size: {embedder.window_size}")
    print(f"Number of Features: {embedder.n_features}")
    print(f"Embedding Dimension: {embedder.embedding_dim}")
    print(f"Device: {embedder.device}")
    print(f"Checkpoint Directory: {embedder.checkpoint_dir}")