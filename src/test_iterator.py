import numpy as np
import tensorflow as tf
import time
import h5py
from Network.PatchHandler3D import PatchHandler3D

def load_indexes(index_file):
    """
        Load patch index file (csv). This is the file that is used to load the patches based on x,y,z index
    """
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

if __name__ == "__main__":
    data_dir = '../data'
    
    # ---- Patch index files ----
    training_file = '{}/train.csv'.format(data_dir)
   
    # Hyperparameters optimisation variables
    epochs =  3
    batch_size = 4

    patch_size = 12
    res_increase = 2
    

    # Load data file and indexes
    trainset = load_indexes(training_file)
    
    # ----------------- TensorFlow stuff -------------------
    # TRAIN dataset iterator
    z = PatchHandler3D(data_dir, patch_size, res_increase, batch_size)
    trainset = z.initialize_dataset(trainset, shuffle=True, n_parallel=2)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        start_time = time.time()
        for i, data_pairs in enumerate(trainset):
            start_loop = time.time()
            
            a = data_pairs
            message = f"Iteration {i+1}   - batc {time.time()-start_loop:.4f} sec {time.time()-start_time:.1f} secs"
            print(f"\r{message}", end='')
            
    print("\nDone")
    