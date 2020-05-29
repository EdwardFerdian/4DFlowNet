import numpy as np
import os
import h5py
import random
import fft_downsampling as fft
import scipy.ndimage as ndimage
from h5functions import save_to_h5

def choose_venc():
    '''
        Give a 68% that data will have a same venc on all 3 velocity components.
    '''
    my_list = ['same'] * 68 + ['diff'] * 32
    return random.choice(my_list)

if __name__ == '__main__':
    # Config
    base_path = '../../data'
    # Put your path to Hires Dataset
    input_filepath  = f'{base_path}/example_data_HR.h5'
    output_filename = f'{base_path}/example_data_LR.h5'
    # Downsample rate
    downsample = 2

    # --- Ready to do downsampling ---
    # setting the seeds for both random and np random, if we need to get the same random order on dataset everytime
    # np.random.seed(10)
    crop_ratio = 1 / downsample
    base_venc_multiplier = 1.1 # Default venc is set to 10% above vmax

    # Possible magnitude and venc values
    mag_values  =  np.asarray([60, 80, 120, 180, 240]) # in px values [0-4095]
    venc_values =  np.asarray([0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]) # in m/s

    # Load the mask once
    with h5py.File(input_filepath, mode = 'r' ) as hf:
        mask = np.asarray(hf['mask'][0])
        data_count = len(hf.get("u"))
    
    is_mask_saved = False # just to mark if the mask already saved or not
    for idx in range(data_count):
        targetSNRdb = np.random.randint(140,170) / 10
        print("Processing data row", idx, "target SNR", targetSNRdb, "db")
        
        # Create the magnitude based on the possible values
        ## This is a part of augmentation to make sure we have varying magnitude
        mag_multiplier = mag_values[idx % len(mag_values)]
        mag_image = mask * mag_multiplier
        
        # Load the velocity U V W from H5
        with h5py.File(input_filepath, mode = 'r' ) as hf:
            mask = np.asarray(hf['mask'][0])

            hr_u = np.asarray(hf['u'][idx])
            hr_v = np.asarray(hf['v'][idx])
            hr_w = np.asarray(hf['w'][idx])
            
            # Calculate the possible VENC for each direction (* 1.1 to avoid aliasing)
            max_u = np.asarray(hf['u_max'][idx]) * base_venc_multiplier
            max_v = np.asarray(hf['v_max'][idx]) * base_venc_multiplier
            max_w = np.asarray(hf['w_max'][idx]) * base_venc_multiplier
        
        # We assume most of the time, we use venc 1.50 m/s
        all_max = np.array([max_u, max_v, max_w])
        
        venc_choice = choose_venc()
        if (venc_choice == 'same'):
            max_vel = np.max(all_max)
            if max_vel < 1.5:
                venc_u = 1.5
                venc_v = 1.5
                venc_w = 1.5
            else:
                # choose a venc up to 2 higher than current max vel
                randindx = np.random.randint(2)
                # print(randindx)
                print('max_vel', max_vel)
                venc = venc_values[np.where(venc_values > max_vel)][randindx]
                venc_u = venc
                venc_v = venc
                venc_w = venc
        else:
            # Different venc
            randindx = np.random.randint(2)
            venc_u = venc_values[np.where(venc_values > max_u)][randindx]

            randindx = np.random.randint(2)
            venc_v = venc_values[np.where(venc_values > max_v)][randindx]

            randindx = np.random.randint(2)
            venc_w = venc_values[np.where(venc_values > max_w)][randindx]
            
            # Skew the randomness by setting main velocity component to 1.5
            main_vel = np.argmax(all_max) # check which one is main vel component
            vencs = [venc_u, venc_v, venc_w]
            if vencs[main_vel] < 1.5:
                print("Forcing venc", main_vel, " to 1.5")
                vencs[main_vel] = 1.5 # just because 1.5 is the common venc

                # set it back to the object
                venc_u = vencs[0]
                venc_v = vencs[1]
                venc_w = vencs[2]
                 
        print(venc_choice, venc_u, venc_v, venc_w)

        # DO the downsampling
        lr_u, mag_u = fft.downsample_phase_img(hr_u, mag_image, venc_u, crop_ratio, targetSNRdb)
        lr_v, mag_v = fft.downsample_phase_img(hr_v, mag_image, venc_v, crop_ratio, targetSNRdb)
        lr_w, mag_w = fft.downsample_phase_img(hr_w, mag_image, venc_w, crop_ratio, targetSNRdb)

        # Save the downsampled images
        save_to_h5(output_filename, "u", lr_u)
        save_to_h5(output_filename, "v", lr_v)
        save_to_h5(output_filename, "w", lr_w)

        save_to_h5(output_filename, "mag_u", mag_u)
        save_to_h5(output_filename, "mag_v", mag_v)
        save_to_h5(output_filename, "mag_w", mag_w)

        save_to_h5(output_filename, "venc_u", venc_u)
        save_to_h5(output_filename, "venc_v", venc_v)
        save_to_h5(output_filename, "venc_w", venc_w)
        save_to_h5(output_filename, "SNRdb", targetSNRdb)
        
        # Only save mask once
        if not is_mask_saved:
            new_mask = ndimage.zoom(mask, crop_ratio, order=1)
            print("Saving downsampled mask...")
            save_to_h5(output_filename, "mask", new_mask)

            is_mask_saved = True


    print("Done!")