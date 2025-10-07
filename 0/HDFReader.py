
import os

import h5py
import pandas as pd
from torch.utils.data import Dataset



class HDF5Reader(Dataset):
    def __init__(self,
                database_path,
                time=False):

        self.path = database_path
        self.current_file = None
        self.available_files = []
        self.current_file_index = 0
        self._scan_directory()

    def _scan_directory(self):
        """
        Scan directory for all .h5 files
        """
        if os.path.isdir(self.path):
            self.available_files = [
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if f.endswith('.h5')
                ]
            if not self.available_files:
                raise FileNotFoundError('No .h5 files found in the specified directory')
            self.available_files.sort()  # Ensure consistent ordering
            self.current_file = self.available_files[0]
        else:
            if not self.path.endswith('.h5'):
                raise ValueError('Single file path must be an .h5 file')
            self.available_files = [self.path]
            self.current_file = self.path

    def next_dataset(self):
        """
        Move to the next dataset in the directory
        """
        if self.current_file_index < len(self.available_files) - 1:
            self.current_file_index += 1
            self.current_file = self.available_files[self.current_file_index]
            return True
        return False

    def get_current_filename(self):
        """
        Get the name of the current file being processed
        """
        return os.path.basename(self.current_file)

    def get_remaining_files(self):
        """
        Get number of remaining files to process
        """
        return len(self.available_files) - self.current_file_index - 1

    def get_total_files(self):
        """
        Get total number of .h5 files
        """
        return len(self.available_files)

    def get_micro(self):
        """
        Converts current H5 file's micro data into a pandas DataFrame.
        The varprim variables are expanded into separate columns.
        Time (microend) is included as the first column.
        
        Returns:
            pd.DataFrame: DataFrame containing the micro time-step data
        """
        with h5py.File(self.current_file, 'r') as f:
            micro_group = f['MACRO']['MICRO']
            varprim_matrix = micro_group['VARPRIM']['varprim'][:]
            microend = micro_group['microend'][:]
            
            df = pd.DataFrame(varprim_matrix, index=microend)
            df.insert(0, 'timestamp', microend)     # Add time as the first column
            
            print(f'File {self.get_current_filename()} has been read and converted into a Pandas DataFrame')
            return df

    def get_macro(self):
        """
        Converts current H5 file's macro data into a pandas DataFrame.
        Time (macroend) is included as the first column.
        
        Returns:
            pd.DataFrame: DataFrame containing the macro time-step data
        """
        with h5py.File(self.current_file, 'r') as f:
            OnlyMACRO_group = f['OnlyMACRO']
            varprim_matrix = OnlyMACRO_group['MACROvarprim'][:]
            macroend = OnlyMACRO_group['macroend'][:]
            
            df = pd.DataFrame(varprim_matrix, index=macroend)
            df.insert(0, 'timestamp', macroend)     # Add time as the first column
            
            print(f'File {self.get_current_filename()} has been read and converted into a Pandas DataFrame')
            return df

    @staticmethod
    def shape(df):
        return df.shape