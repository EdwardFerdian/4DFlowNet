import numpy as np
import tensorflow as tf
import time
import h5py
from Network.PatchHandler3D import PatchHandler3D
from Network.TrainerSetup import TrainerSetup

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
    validate_file = '{}/validate.csv'.format(data_dir)

    QUICKSAVE = True
    benchmark_file = '{}/benchmark.csv'.format(data_dir)
    
    # Hyperparameters optimisation variables
    initial_learning_rate = 1e-4
    epochs =  1000
    batch_size = 20
    mask_threshold = 0.6

    # Network setting
    network_name = '4DFlowNet'
    patch_size = 16
    res_increase = 2
    # Residual blocks, default (8 LR ResBlocks and 4 HR ResBlocks)
    low_resblock = 8
    hi_resblock = 4

    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset = load_indexes(validate_file)
    
    # ----------------- TensorFlow stuff -------------------
    # Reset all the tensor variables
    tf.reset_default_graph()  

    # TRAIN dataset iterator
    z = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    iterator = z.initialize_dataset(trainset, training=True)

    # VALIDATION iterator
    valdh = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    val_iterator = valdh.initialize_dataset(valset, training=False)

    # Bechmarking iterator, use to keep track of prediction progress per best model
    benchmark_iterator = None
    if QUICKSAVE and benchmark_file is not None:
        benchmark_set = load_indexes(benchmark_file)
        ph = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
        benchmark_iterator = ph.initialize_dataset(benchmark_set, training=False)

    # ------- Main Network ------
    print(f"4DFlowNet Patch {patch_size}, lr {initial_learning_rate}, batch {batch_size}")
    network = TrainerSetup(patch_size, res_increase, initial_learning_rate, QUICKSAVE, network_name, benchmark_iterator, low_resblock, hi_resblock)
    session = network.init_model_dir()
    network.train_network(train_iterator=iterator, val_iterator=val_iterator, n_epoch=epochs)


    