
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler



class DataPreprocessor:
    """
    Preprocessing class for the raw data:
    - Remove NaNs and duplicates
    - Scale the data using MinMax or Standard scaler (scaler saved to - scalerpath -)

    Args:
        - database: DataFrame containing the raw data
        - method: string, either "MinMax", "Standard" or "Robust"
        - scalerpath: path to save/load the scaler
    """
    def __init__(self, database, method, scalerpath):
        self.database = database
        self.method = method
        self.scalerpath = scalerpath
        self.features_scaler = None
        self.mean_ = None
        self.std_ = None
        
        # Separate timestamp and features
        self.timestamp = database['timestamp'].values
        self.features = database.drop('timestamp', axis=1)
    
    def encode_time(self, time_values):
        """
        Encodes numeric timestamps into 15 features:
        - Relative position (1 feature)
        - Multi-scale differences (3 features)
        - Fourier features (8 features)
        - Position encoding (3 features)
        """
        # Ensure we're working with numpy array
        time_values = np.asarray(time_values)
        n_samples = len(time_values)
        
        # 1. Relative position (0 to 1)
        time_range = time_values.max() - time_values.min()
        relative_pos = (time_values - time_values.min()) / (time_range if time_range != 0 else 1)
        
        # 2. Multi-scale differences
        # Calculate differences between consecutive timestamps
        diffs = np.diff(time_values, prepend=time_values[0])
        diffs[0] = diffs[1]  # Fix first value
        
        # Compute different scales of differences
        mean_diff = np.mean(np.abs(diffs))
        if mean_diff == 0:
            mean_diff = 1
        
        p75 = np.percentile(np.abs(diffs), 75)
        if p75 == 0:
            p75 = 1
            
        p99 = np.percentile(np.abs(diffs), 99)
        if p99 == 0:
            p99 = 1
        
        diff_features = np.column_stack([
            diffs / mean_diff,  # Local scale
            diffs / p75,        # Medium scale
            diffs / p99         # Global scale
        ])
        
        # 3. Fourier features
        # Generate 4 different frequencies
        freqs = np.array([1, 2, 4, 8])
        angles = 2 * np.pi * np.outer(relative_pos, freqs)
        fourier_features = np.column_stack([
            np.sin(angles),
            np.cos(angles)
        ])
        
        # 4. Position encoding
        positions = np.arange(n_samples)
        position_features = np.column_stack([
            positions / n_samples,
            np.sin(2 * np.pi * positions / n_samples),
            np.cos(2 * np.pi * positions / n_samples)
        ])
        
        # Combine all features
        time_features = np.column_stack([
            relative_pos,          # 1 feature
            diff_features,         # 3 features
            fourier_features,      # 8 features
            position_features      # 3 features
        ])
        
        # Normalize features
        if self.mean_ is None or self.std_ is None:
            self.mean_ = np.mean(time_features, axis=0)
            self.std_ = np.std(time_features, axis=0)
            self.std_[self.std_ == 0] = 1  # Avoid division by zero
        
        time_features = (time_features - self.mean_) / self.std_
        
        return time_features
    
    def __clean__(self):
        """Clean the DataFrame by removing NaNs and Duplicates."""
        # Create a copy to avoid modifying the original
        df = self.database.copy()
        
        # Remove NaN values
        df_no_nan = df.dropna()
        print(f"Removed {len(df) - len(df_no_nan)} NaN values")
        
        # Remove duplicates
        df_clean = df_no_nan.drop_duplicates()
        print(f"Removed {len(df_no_nan) - len(df_clean)} duplicates")
        
        # Update class attributes with cleaned data
        self.database = df_clean
        self.timestamp = df_clean['timestamp'].values
        self.features = df_clean.drop('timestamp', axis=1)
        
        return self

    def __transform__(self):
        """Transform data with separate handling for time and other features."""
        if not os.path.exists(self.scalerpath):
            os.makedirs(self.scalerpath)
            
        scaler_filename = f"{self.method}Scaler.pkl"
        scaler_path = os.path.join(self.scalerpath, scaler_filename)
        
        # Handle non-time features
        if os.path.exists(scaler_path):
            self.features_scaler = joblib.load(scaler_path)
            print(f"{self.method}Scaler loaded successfully")
        else:
            if self.method == "Robust":
                self.features_scaler = RobustScaler()
            elif self.method == "MinMax":
                self.features_scaler = MinMaxScaler()
            else:
                self.features_scaler = StandardScaler()
            self.features_scaler.fit(self.features)
            joblib.dump(self.features_scaler, scaler_path)
            print(f"{self.method}Scaler created and saved")
        
        # Transform features using the cleaned data's index
        scaled_features = pd.DataFrame(
            self.features_scaler.transform(self.features),
            columns=self.features.columns,
            index=self.features.index  # Use the cleaned data's index
        )
        
        # Encode time
        time_features = self.encode_time(self.timestamp)
        time_columns = [f't{i}' for i in range(time_features.shape[1])]
        encoded_time = pd.DataFrame(
            time_features,
            columns=time_columns,
            index=self.features.index  # Use the cleaned data's index
        )
        
        # Combine all features
        result = pd.concat([encoded_time, scaled_features], axis=1)
        print("\nTransformed data summary:")
        print(f"Time features shape: {encoded_time.shape}")
        print(f"Scaled features shape: {scaled_features.shape}")
        print(f"Final shape: {result.shape}")
        
        return result
    
    def __invTransform__(self, scaled_data):
        """Inverse transform scaled data, handling time features separately."""
        if self.features_scaler is None:
            raise ValueError("Scaler not found. Run transform() first.")

        # Convert input to 2D array if needed
        if len(scaled_data.shape) == 1:
            scaled_data = scaled_data.reshape(1, -1)

        # For embedded data
        if scaled_data.shape[1] == 256:  # Embedding dimension
            return pd.DataFrame(
                self.features_scaler.inverse_transform(scaled_data),
                columns=self.features.columns
            )
        
        # For regular data
        # Separate time features and regular features
        n_time_features = 15  # Fixed number of time features
        time_features = scaled_data[:, :n_time_features]
        other_features = scaled_data[:, n_time_features:]
        
        # Inverse transform only the non-time features
        original_features = pd.DataFrame(
            self.features_scaler.inverse_transform(other_features),
            columns=self.features.columns
        )
        
        # Denormalize time features and reconstruct original timestamp
        time_features = time_features * self.std_[:n_time_features] + self.mean_[:n_time_features]
        reconstructed_time = self.timestamp.min() + time_features[:, 0] * (self.timestamp.max() - self.timestamp.min())
        
        time_df = pd.DataFrame(reconstructed_time, columns=['timestamp'])
        
        return pd.concat([time_df, original_features], axis=1)

    def process(self):
        """Execute the complete preprocessing pipeline."""
        return self.__clean__().__transform__()