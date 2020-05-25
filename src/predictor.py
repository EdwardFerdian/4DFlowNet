import tensorflow as tf
import numpy as np
import time
from Network.SRLoader import SRLoader
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.ImageDataset import ImageDataset

if __name__ == '__main__':
    input_dir = '../data'
    filename = 'example_data.h5'
    
    output_dir = "../result"
    output_filename = 'SR_result.h5'
    
    model_dir = "../models/4DFlowNet"
    # Params
    patch_size = 24
    res_increase = 2
    batch_size = 8
    round_small_values = True

    # Setting up
    input_filepath = '{}/{}'.format(input_dir, filename)
    pgen = PatchGenerator(patch_size, res_increase)
    dataset = ImageDataset()

    # Check the number of rows in the file
    nr_rows = dataset.get_dataset_len(input_filepath)
    print(f"Number of rows in dataset: {nr_rows}")

    print(f"Loading 4DFlowNet: {res_increase}x upsample")
    # Load the network
    network = SRLoader(model_dir, patch_size, res_increase)

    # loop through all the rows in the input file
    for nrow in range(0, nr_rows):
        print("\n--------------------------")
        print(f"\nProcessing ({nrow+1}/{nr_rows}) - {time.ctime()}")
        # Load data file and indexes
        dataset.load_vectorfield(input_filepath, nrow)
        print(f"Original image shape: {dataset.u.shape}")
        
        velocities, magnitudes = pgen.patchify(dataset)
        data_size = len(velocities[0])
        print(f"Patchified. Nr of patches: {data_size} - {velocities[0].shape}")

        # Predict the patches
        results = np.zeros((0,patch_size*res_increase, patch_size*res_increase, patch_size*res_increase, 3))
        start_time = time.time()

        for current_idx in range(0, data_size, batch_size):
            time_taken = time.time() - start_time
            print(f"\rProcessed {current_idx}/{data_size} Elapsed: {time_taken:.2f} secs.", end='\r')
            # Prepare the batch to predict
            patch_index = np.index_exp[current_idx:current_idx+batch_size]
            sr_images = network.predict(velocities[0][patch_index],
                                    velocities[1][patch_index],
                                    velocities[2][patch_index],
                                    magnitudes[0][patch_index],
                                    magnitudes[1][patch_index],
                                    magnitudes[2][patch_index])

            results = np.append(results, sr_images, axis=0)
        # End of batch loop    
        time_taken = time.time() - start_time
        print(f"\rProcessed {data_size}/{data_size} Elapsed: {time_taken:.2f} secs.")

        # Denormalized the prediction
        results = dataset.postprocess_result(results, zerofy=round_small_values)

        # Reconstruct the image from the patches
        predictions = pgen.unpatchify(results)
        print(f"Image reconstructed: {predictions.shape}")

        # Prepare to save
        predictions = np.expand_dims(predictions, axis=0) 
        prediction_utils.save_predictions(output_dir, output_filename, dataset.velocity_colnames, predictions, compression='gzip')
        # Save spacing if the original info was there
        if dataset.dx is not None:
            new_spacing = dataset.dx / res_increase
            new_spacing = np.expand_dims(new_spacing, axis=0) 
            prediction_utils.save_to_h5(f'{output_dir}/{output_filename}', dataset.dx_colname, new_spacing, compression='gzip')

    print("Done!")