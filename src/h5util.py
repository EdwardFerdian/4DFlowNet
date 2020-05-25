import os
import h5py
import numpy as np

def save_predictions(output_path, output_filename, col_name, dataset, compression=None):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    output_filepath = os.path.join(output_path, output_filename)

    # convert float64 to float32 to save space
    if dataset.dtype == 'float64':
        dataset = np.array(dataset, dtype='float32')
    
    with h5py.File(output_filepath, 'a') as hf:    
        if col_name not in hf:
            datashape = (None, )
            if (dataset.ndim > 1):
                datashape = (None, ) + dataset.shape[1:]
            hf.create_dataset(col_name, data=dataset, maxshape=datashape, compression=compression) # gzip, compression_opts=4
        else:
            hf[col_name].resize((hf[col_name].shape[0]) + dataset.shape[0], axis = 0)
            hf[col_name][-dataset.shape[0]:] = dataset
