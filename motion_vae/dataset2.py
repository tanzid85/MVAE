# import scipy.io
# import os
# def load_all_mat_files(directory):
#     mat_files_data = {}
#     for filename in os.listdir(directory):
#         if filename.endswith('.mat'):
#             file_path = os.path.join(directory, filename)
#             data = scipy.io.loadmat(file_path)
#             mat_files_data[filename] = data
#     return mat_files_data
#
# # Replace this with the path to the directory containing your .mat files
# directory_path = '/home/mdtanzid/PycharmProjects/MVAE/Data'
# all_data = load_all_mat_files(directory_path)
#
# # Print all file names
# print("Loaded .mat files:")
# for filename in all_data.keys():
#     print(filename)
#
# # Optionally, print the content of a specific file
# specific_file = 'trajectory_14_obs.mat'  # Replace with a file name of interest
# if specific_file in all_data:
#     print(f"Contents of {specific_file}:")
#     for key, value in all_data[specific_file].items():
#         print(f"{key}: {value}")
#
# x =1
#
# # Now all_data is a dictionary where each key is the filename of a .mat file
# # and each value is the content of that .mat file

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
import random
import copy
import os
from utils.torch_transform import rotmat_to_rot6d
from motion_vae.config import *

class MockOptions:
    def __init__(self):
        self.num_condition_frames = 1


class TimeSeriesPoseDataset(Dataset):
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)

        # Load data from multiple .mat files
        directory = '/home/mdtanzid/PycharmProjects/MVAE/Data'
        self.trajectories = []  # List to store each trajectory separately

        for filename in os.listdir(directory):
            if filename.endswith('.mat'):
                file_path = os.path.join(directory, filename)
                mat_data = loadmat(file_path)
                joint_pos = mat_data['obs_data'][:, :16]
                joint_velo = mat_data['obs_data'][:, 16:32]
                self.trajectories.append((joint_pos, joint_velo))

        self.std, self.avg = None, None
        # Calculate normalization stats if needed
        self.get_normalization_stats()  # Uncomment if normalization stats need to be calculated

    # def get_normalization_stats(self):

    # Implement normalization stats calculation across all trajectories
    # ...

    def __len__(self):
        # Total number of sequences across all trajectories
        return sum(max(0, len(joint_pos) - self.opt.num_condition_frames) for joint_pos, _ in self.trajectories)

    def __getitem__(self, idx):
        # Find which trajectory this index belongs to
        for joint_pos, joint_velo in self.trajectories:
            if idx < len(joint_pos) - self.opt.num_condition_frames:
                break
            idx -= len(joint_pos) - self.opt.num_condition_frames

        joint_pos_seq = joint_pos[idx:idx + self.opt.num_condition_frames]
        joint_velo_seq = joint_velo[idx:idx + self.opt.num_condition_frames]
        feature = np.concatenate([joint_pos_seq, joint_velo_seq], axis=1)

        # Normalize features if normalization stats are set
        if self.std is not None:
            feature = (feature - self.avg) / self.std

        data_dict = {
            'feature': feature,
            'start': idx,
        }
        return data_dict

def test_dataset():
    opt = MockOptions()
    dataset = TimeSeriesPoseDataset(opt)

    print(f"Total sequences: {len(dataset)}")

    for i in range(min(5, len(dataset))):
        item = dataset[i]
        print(f"Item {i}: Feature shape: {item['feature'].shape}, Start: {item['start']}")

if __name__ == "__main__":
    test_dataset()
    x = 1