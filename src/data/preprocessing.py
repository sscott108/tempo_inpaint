import pickle
import torch
def load_classification_pickle(input_file):
    """Load from pickle"""
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    return data["complete"], data["partial"], data["blank"]


# def custom_collate_fn(batch):
#     """Custom collate function to handle variable-length station names"""
    
#     # Separate the station_names from other tensor data
#     collated_batch = {}
    
#     # Handle tensor data normally
#     tensor_keys = ['p_mask', 'p_val_mask', 'masked_img', 'known_mask', 'target']
#     if 'known_and_fake_mask' in batch[0]:
#         tensor_keys.append('known_and_fake_mask')
#     if 'fake_mask' in batch[0]:
#         tensor_keys.append('fake_mask')
    
#     for key in tensor_keys:
#         collated_batch[key] = torch.stack([item[key] for item in batch])
    
#     # Handle paths as list
#     collated_batch['path'] = [item['path'] for item in batch]
    
#     # Handle station_names as list of lists (don't try to stack)
#     collated_batch['station_names'] = [item['station_names'] for item in batch]
    
#     return collated_batch

def custom_collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            # Keep as list so we can index it later
            out[k] = [b[k] for b in batch]
    return out