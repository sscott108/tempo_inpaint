import pickle

def load_classification_pickle(input_file):
    """Load from pickle"""
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    return data["complete"], data["partial"], data["blank"]