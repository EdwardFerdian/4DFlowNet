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
    initial_learning_rate = 1e-4
    epochs =  2
    batch_size = 20
    mask_threshold = 0.6

    # Network setting
    patch_size = 16
    res_increase = 2

    # Load data file and indexes
    trainset = load_indexes(training_file)
    
    # ----------------- TensorFlow stuff -------------------
    # Reset all the tensor variables
    tf.reset_default_graph()  

    # TRAIN dataset iterator
    z = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    iterator = z.initialize_dataset(trainset, training=True)

    # Loop the iterator
    next_element = iterator.get_next()
    session = tf.Session()

    session.run(iterator.initializer)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        start_time = time.time()
        try:
            i = 0
            while True:
                start_loop = time.time()
                next_batch = session.run(next_element)
                message = f"Iteration {i+1}   - batc {time.time()-start_loop:.4f} sec {time.time()-start_time:.1f} secs"
                print(f"\r{message}", end='')
                i+=1 

        except tf.errors.OutOfRangeError:
            session.run(iterator.initializer)
            pass