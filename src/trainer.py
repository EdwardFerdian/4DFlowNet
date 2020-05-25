import numpy as np
import tensorflow as tf
import time
import h5py
from Network.PatchHandler3D import PatchHandler3D
from Network.TrainerSetup import TrainerSetup

def load_indexes(index_file):
    indexes = np.genfromtxt(index_file, delimiter=',', skip_header=True, dtype='unicode') # 'unicode' or None
    return indexes

if __name__ == "__main__":
    # data_dir = '../data'
    data_dir = r'C:\Users\efer502\Desktop\test-ansys\h5\aorta'
    
    # ---- Patch index files ----
    training_file = '{}/abc16train.csv'.format(data_dir)
    validate_file = '{}/abc16val.csv'.format(data_dir)

    QUICKSAVE = False
    benchmark_file = None
    
    # Hyperparameters optimisation variables
    patch_size = 16
    res_increase = 2

    initial_learning_rate = 1e-4
    epochs =  3
    batch_size = 20
    mask_threshold = 0.6

    network_name = '4DFlow-test'

    # Load data file and indexes
    trainset = load_indexes(training_file)
    valset = load_indexes(validate_file)
    
    # ----------------- TensorFlow stuff -------------------
    # Reset all the tensor variables
    tf.reset_default_graph()  

    # the data pipeline will zoom the data
    z = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    iterator = z.initialize_dataset(trainset, training=True)

    # VALIDATION iterator
    valdh = PatchHandler3D(data_dir, patch_size, res_increase, batch_size, mask_threshold)
    val_iterator = valdh.initialize_dataset(valset, training=False)

    # ------- Main Network ------
    print('\nBuilding network model...')

    print("4DFlowNet Patch {}, lr {}, batch {}".format(patch_size, initial_learning_rate, batch_size))
    network = TrainerSetup(patch_size, res_increase, initial_learning_rate, quicksave_enable=QUICKSAVE, network_name=network_name)
    session = network.init_model_dir()
    network.train_network(train_iterator=iterator, val_iterator=val_iterator, n_epoch=epochs)

    