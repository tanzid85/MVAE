import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
import random
import copy
from utils.torch_transform import rotmat_to_rot6d
from motion_vae.config import *

class TimeSeriesPoseDataset(Dataset):
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)

        # Assuming your .mat file contains joint_pos in the first 16 columns and joint_velo in the next 16 columns
        mat_data = loadmat('/home/mdtanzid/PycharmProjects/MVAE/Old Data/tot_obs.mat')
        self.joint_pos = mat_data['obs_data'][:, :16]
        self.joint_velo = mat_data['obs_data'][:, 16:32]

        # self.valid_arr = len(self.joint_pos)
        # self.sequences = []
        # self.selected_arr = np.zeros_like(self.valid_arr)
        # if opt.predict_phase:
        #     self.phase_arr = np.zeros((self.valid_arr, 2), dtype=float)
        #     self.phase_rad_arr = np.zeros(self.valid_arr, dtype=float)

        # self.sequence_length = 2  # Define the length of each sequence
        self.total_sequences = len(self.joint_pos) - self.sequence_length + 1
        self.std, self.avg = None, None
        self.sequences = 2

        self.seq_weights = np.array(
            [seq['length'] for seq in self.sequences], dtype=float)
        self.seq_weights /= np.sum(self.seq_weights)

    def get_normalization_stats(self):
        opt = self.opt
        feature_all = None
        feature_all = np.concatenate([self.joint_pos, self.joint_velo], axis=1)

        # Adjust the normalization logic based on your features and requirements
        std = np.std(feature_all, axis=0)
        std[std == 0] = 1.0
        avg = np.average(feature_all, axis=0)
        self.std = std
        self.avg = avg

    def set_normalization_stats(self, avg, std):
        self.avg = avg
        self.std = std

    def __len__(self):
        return self.opt.nseqs

    def __getitem__(self, idx):
        # Ensure the index is within the range of total sequences
        idx = idx % self.total_sequences

        joint_pos_seq = self.joint_pos[idx:idx + self.sequence_length]
        joint_velo_seq = self.joint_velo[idx:idx + self.sequence_length]

        # Normalize features if needed
        feature = np.concatenate([joint_pos_seq, joint_velo_seq], axis=1)
        if self.std is not None:
            feature = (feature - self.avg) / self.std

        data_dict = {
            'feature': feature,
            'start': idx,
        }
        return data_dict

    def sample_first_frame(self):
        opt = self.opt
        data = self.__getitem__(0)
        T = opt.num_condition_frames
        # Extract the first frame features from your data matrices
        frame = {
            'joint_pos': self.joint_pos[data['start'] + T, :],
            'joint_velo': self.joint_velo[data['start'] + T, :],
            # Include other features as needed
        }
        return frame

# To test the dataset.py file you can use the code below.
dataset = TimeSeriesPoseDataset(opt)
dataset.get_normalization_stats()

# Now, access the calculated statistics directly from the dataset object
# print(f"Normalization mean: {dataset.avg}")
# print(f"Normalization standard deviation: {dataset.std}")

trainset = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=opt.num_threads
)

# # Assuming the DataLoader is already created and named as 'trainset'
# for i, batch in enumerate(trainset):
#     print(f"Batch {i}:")
#     print(batch)  # This will print the entire batch. You can also access specific parts of the batch.

#     # To limit the number of batches printed, you can add a condition to break the loop
#     if i == 0:  # This will print the first two batches (0 and 1)
#         break

for i, batch_data in enumerate(trainset):
    # Check only the first batch
    if i == 0:
        # Extract features tensor from the batch
        features = batch_data['feature']

        # Print the shape of the features tensor
        print("Features shape:", features.shape)

        # Print the size of the batch (number of sequences in the batch)
        print("Batch size:", len(features))

        # Optionally, print the start indices
        start_indices = batch_data['start']
        print("Start indices of sequences in batch:", start_indices)
        break
