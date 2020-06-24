import numpy as np
import pickle
import torch
from scipy import ndimage

from torch.utils.data import Dataset

class NusceneDataset(Dataset):
    """Nuscene Dataset for prediction"""
    def __init__(self, dills, max_A):
        """
        :dills: dill file list
        """
        self.dills = dills
        self.max_A = max_A
        
    def __len__(self):
        return len(self.dills)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        def fetch_item(data, single_idx):
            with open(data[single_idx], 'rb') as f:
                return pickle.load(f)
        
        data = fetch_item(self.dills, idx)
        
        # others
        def fill_axis_to_size(arr, axis, size, fill=0., clip=False):
            if arr.shape[axis] > size:
                if clip:
                    return np.take(arr, axis=axis, indices=range(0, size))
                else:
                    raise ValueError("Axis too large!")
            elif arr.shape[axis] == size:
                return arr
            else:
                diff = size - arr.shape[axis]
                new_shape = list(arr.shape)
                new_shape[axis] = diff
                return np.concatenate((arr, fill * np.ones(new_shape, dtype=np.float32)), axis=axis)
            
        Other_count = self.max_A - 1
        other_experts = np.stack(
            fill_axis_to_size(np.asarray(data['agent_futures'][...,:2])[:Other_count], axis=0, size=Other_count)
        )
        
        other_pasts = np.stack(
            fill_axis_to_size(np.asarray(data['agent_pasts'][...,:2])[:Other_count], axis=0, size=Other_count)
        )
        
        other_yaws = np.stack(
            fill_axis_to_size(np.asarray(data['agent_yaws'])[:Other_count], axis=0, size=Other_count)
        )
        
        agent_presence = np.zeros((self.max_A), dtype=np.float32)
        if isinstance(data['agent_futures'], np.ndarray):
            size = data['agent_futures'].shape[0]
        else:
            size = len(data['agent_futures'])
        agent_presence[:size+1] = 1
        
        
        overhead_features = data['overhead_features']  # (H, W, C)
        
        # Signed distance transform
        signed_distrance_transformed_images = []
        for i in range(5):
            binary_image = np.array(overhead_features[..., i] > 5/255.)
            dt = ndimage.distance_transform_edt
            sdt = (dt(binary_image) - dt(1 - binary_image))
            signed_distrance_transformed_images.append(sdt)
            
        overhead_sdt_features = np.concatenate(
            np.array(signed_distrance_transformed_images)[..., np.newaxis], axis=2)
        
        cliped_sdt = np.clip(overhead_sdt_features, -10, 1)
        normalized_sdt = (cliped_sdt + 4.5) / 5.5
        
        sample = {
            'player_expert': data['player_future'][...,:2], 
            'player_past': data['player_past'][...,:2],
            'player_yaw': data['player_yaw'],
            'overhead_features': np.transpose(data['overhead_features'], (2, 0, 1)),
            'overhead_sdt_features': np.transpose(normalized_sdt, (2, 0, 1)),
            'other_experts': other_experts,
            'other_pasts': other_pasts,
            'other_yaws': other_yaws,
            'agent_presence': agent_presence,
            'idx': idx
        }
        
        return sample