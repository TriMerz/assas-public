# config.py
import argparse
import yaml
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    # Required arguments (no default values) first
    database_path: str
    model_name: str
    window_size: int
    use_embeddings: bool
    is_macro: bool
    default_encoder: bool

    # Optional arguments (with default values) after
    scalerpath: str = ""
    checkpoint_dir: str = ""
    new_test: bool = False
    test_number: int = 0
    embedding_dim: int = 256
    num_layers: int = 3        # Nuovo parametro
    layer_dim: int = 1024      # Nuovo parametro
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    validation_split: float = 0.2
    weight_decay: float = 1e-4
    patience: int = 20
    data_augmentation: bool = False
    scaler_method: str = "MinMax"
    time_encoding_dim: int = 64
    conv_channels: int = 128
    n_heads: int = 8
    dropout_rate: float = 0.1
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Create config from YAML file with proper boolean parsing"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Explicitly convert boolean strings to actual booleans
        bool_fields = ['use_embeddings', 'is_macro', 'new_test', 
                      'data_augmentation', 'default_encoder']
        
        for field in bool_fields:
            if field in config_dict:
                if isinstance(config_dict[field], str):
                    config_dict[field] = config_dict[field].lower() == 'true'
                elif isinstance(config_dict[field], (int, float)):
                    config_dict[field] = bool(config_dict[field])
                    
        # Print the parsed values for debugging
        print("\nDEBUG: Parsed configuration values:")
        for key, value in config_dict.items():
            if key in bool_fields:
                print(f"{key}: {value} (type: {type(value)})")
                
        return cls(**config_dict)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration parameters and return status and error messages."""
        messages = []
        
        # Check required paths
        if not self.database_path:
            messages.append("Error: database_path is required")
        elif not os.path.exists(self.database_path):
            messages.append(f"Error: database_path does not exist: {self.database_path}")
            
        # Validate model name
        valid_models = ["CNN", "FNO", "GraFITi", "MLP", "RNN", "Transformer"]
        if not self.model_name:
            messages.append("Error: model_name is required")
        elif self.model_name not in valid_models:
            messages.append(f"Error: model_name must be one of {valid_models}")
            
        # Validate numeric parameters
        if self.window_size <= 0:
            messages.append("Error: window_size must be positive")
        if self.embedding_dim <= 0:
            messages.append("Error: embedding_dim must be positive")
        if self.epochs <= 0:
            messages.append("Error: epochs must be positive")
        if self.batch_size <= 0:
            messages.append("Error: batch_size must be positive")
        if not (0 < self.learning_rate < 1):
            messages.append("Error: learning_rate must be between 0 and 1")
        if not (0 <= self.validation_split < 1):
            messages.append("Error: validation_split must be between 0 and 1")
        
        # Validate new parameters
        if self.num_layers <= 0:
            messages.append("Error: num_layers must be positive")
        if self.layer_dim <= 0:
            messages.append("Error: layer_dim must be positive")
            
        # Validate scaler method
        valid_scalers = ["MinMax", "Standard", "Robust"]
        if self.scaler_method not in valid_scalers:
            messages.append(f"Error: scaler_method must be one of {valid_scalers}")
            
        return len(messages) == 0, messages

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Processing and Embedding')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    # Rimuoviamo i default values e rendiamo tutti gli argomenti opzionali
    parser.add_argument('--database_path', type=str,
                       help='Path to HDF5 database directory or file')
    
    parser.add_argument('--model_name', type=str, 
                       choices=["CNN", "FNO", "GraFITi", "MLP", "RNN", "Transformer"],
                       help='Name of the model to use')
    
    parser.add_argument('--window_size', type=int,
                       help='Size of the sliding window')
    
    parser.add_argument('--use_embeddings', type=str, 
                       choices=['true', 'false'],
                       help='Whether to use embeddings (true/false)')
    
    parser.add_argument('--is_macro', type=str,
                       choices=['true', 'false'],
                       help='Use MACRO (true) or MICRO (false) data')
    
    # Questo era il problema! action='store_true' imposta un default a False
    # Cambiamolo per non avere default
    parser.add_argument('--new_test', type=str,
                       choices=['true', 'false'],
                       help='Create new test directory')
    
    parser.add_argument('--test_number', type=int,
                       help='Specific test number to use (if not new_test)')
    
    parser.add_argument('--scaler_method', type=str,
                       choices=["MinMax", "Standard", "Robust"],
                       help='Method to use for scaling data')
    
    args = parser.parse_args()
    
    # Convert string booleans to actual booleans only if they are provided
    if args.use_embeddings is not None:
        args.use_embeddings = args.use_embeddings.lower() == 'true'
    if args.is_macro is not None:
        args.is_macro = args.is_macro.lower() == 'true'
    if args.new_test is not None:
        args.new_test = args.new_test.lower() == 'true'
        
    return args

def setup_config():
    """Setup configuration from command line args and config file"""
    args = parse_args()
    
    # Load base config from YAML
    config = ModelConfig.from_yaml(args.config)
    
    # Only override if explicitly provided in command line
    for arg in vars(args):
        if getattr(args, arg) is not None and arg != 'config':
            current_value = getattr(config, arg)
            new_value = getattr(args, arg)
            if current_value != new_value:
                print(f"\nOverriding {arg}: {current_value} -> {new_value}")
                setattr(config, arg, new_value)
    
    return config

def create_test_directory(model_name: str, base_path: str, test_number: int = None) -> str:
    """Create a new test directory with incremental numbering"""
    if test_number is not None:
        test_dir = os.path.join(base_path, f"{model_name}_{test_number}")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        return test_dir
    
    test_num = 0
    while True:
        test_dir = os.path.join(base_path, f"{model_name}_{test_num}")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            return test_dir
        test_num += 1