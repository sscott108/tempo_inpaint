import pickle

def load_classification_pickle(input_file="/hpc/home/srs108/TEMPO/file_classification_nop.pkl"):
    """Load from pickle"""
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    return data["complete"], data["partial"], data["blank"]