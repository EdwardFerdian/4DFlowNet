# 4DFlowNet
Super Resolution 4D Flow MRI using Residual Neural Network

This is an implementation of the paper [4DFlowNet: Super-Resolution 4D Flow MRI](https://www.frontiersin.org/articles/10.3389/fphy.2020.00138/full) using Tensorflow 2.2.0 with Keras.

## Updates 4DFlowNet v2.0
- Loss function has been updated (MSE fluid + MSE non fluid)
- L2 regularization added
- Divergence loss turned off
- Evaluation metrics updated
- Final activation layer switch to linear to allow phase aliasing

These changes are implemented for [Cerebrovascular super-resolution 4D Flow MRI](https://www.biorxiv.org/content/10.1101/2021.08.25.457611v1.full)

## Manuscript version

Original network implementation from the manuscript can be found under the following branches:
- release/manuscript_version (for TF2.0)
- tf1.8 (for Tensorflow 1.8.0)

The pre-trained networks weights can be found here:

- [Original 4DFlowNet pre-trained weights](https://auckland.figshare.com/articles/Super_Resolution_4DFlow_MRI/12253424)

- [Cerebrovascular 4DFlowNet weights](https://auckland.figshare.com/articles/software/Cerebrovascular_4DFlowNet_-_Super_Resolution_4D_Flow_MRI/19158122)

Training dataset is available for download from:
- [Aortic CFD dataset](https://auckland.figshare.com/articles/dataset/4DFlowNet_-_high_resolution_aortic_CFD_dataset/24424888)

# Example results

Below are the example prediction results from an actual 4D Flow MRI of a bifurcation phantom dataset. 

LowRes input (voxel size 4mm)
<p align="left">
    <img src="https://i.imgur.com/O48FbAh.gif" width="330">
</p>

High Res Ground Truth vs noise-free Super Resolution (2mm)
<p align="left">
    <img src="https://i.imgur.com/67CRdGn.gif" width="350">
</p>

High Res Ground Truth vs noise-free Super Resolution (1mm)
<p align="left">
    <img src="https://i.imgur.com/DMQa2Lr.gif" width="350">
</p>


# Training setup from CFD data
## Prepare dataset

To prepare training or validation dataset, we assume a High resolution CFD  dataset is available. As an example we have provided this under /data/example_data_HR.h5

How to prepare training/validation dataset.

    1. Generate lowres dataset
        >> Configure the datapath and filenames in prepare_lowres_dataset.py
        >> Run prepare_lowres_dataset.py
        >> This will generate a separate HDF5 file for the low resolution velocity data.
    2. Generate random patches from the LR-HR dataset pairs.
        >> Configure the datapath and filenames in prepare_patches.py
        >> Configure patch_size, rotation option, and number of patches per frame
        >> Run prepare_patches.py
        >> This will generate a csv file that contains the patch information.

## Training

The training accepts csv files for training and validation set. A benchmark set is used to keep prediction progress everytime a model is being saved as checkpoint. Example csv files are provided in the /data folder.

To run a training for 4DFlowNet:

    1. Put all data files (HDF5) and CSV patch index files in the same directory (e.g. /data)
    2. Open trainer.py and configure the data_dir and the csv filenames
    3. Adjust hyperparameters. The default values from the paper are already provided in the code.
    4. Run trainer.py

Adjustable parameters:

|Param  | Description   | Default|
|------|--------------|--------:|
| patch_size| The image will be split into isotropic patches. Adjust according to computation power and image size.  | 24|
| res_increase| Upsample ratio. Adjustable to any integer. More upsample ratio requires more computation power. *Note*: res_increase=1 will denoise the image at the current resolution |2|
| batch_size| Batch size per prediction. Keep it low. |8|
| initial_learning_rate| Initial learning rate |1e-4|
| epochs | maximum number of epochs | 1000 |
| mask_threshold| Mask threshold for non-binary mask. This is used to measure relative error (accuracy) | 0.6 |
| network_name | The network name. The model will be saved in this name_timestamp format |4DFlowNet|
|QUICKSAVE| Option to run a "bechmark" dataset everytime a model is saved | True |
|benchmark_file| A patch index file (CSV) contains a list of patches. Only the first batch will be read and run into prediction. | None|
| low_resblock | Number of residual blocks in low resolution space within 4DFlowNet. |8|
| hi_resblock | Number of residual blocks in high resolution space within 4DFlowNet. |4|



# Running prediction on MRI data
## Prepare data from MRI (for prediction purpose)

To prepare 4D Flow MRI data to HDF5, go to the prepare_data/ directory and run the following script:

    >> python prepare_data.py --input-dir [4DFlowMRI_CASE_DIRECTORY]

    >> usage: prepare_mri_data.py [-h] --input-dir INPUT_DIR
                           [--output-dir OUTPUT_DIR]
                           [--output-filename OUTPUT_FILENAME]
                           [--phase-pattern PHASE_PATTERN]
                           [--mag-pattern MAG_PATTERN] [--fh-mul FH_MUL]
                           [--rl-mul RL_MUL] [--in-mul IN_MUL]

Notes: 
*  The directory must contains the following structure:
    [CASE_NAME]/[Magnitude_or_Phase]/[TriggerTime]
* There must be exactly 3 Phase and 3 Magnitude directories 
* To get the required directory structure, [DicomSort](https://dicomsort.com/) is recommended. Sort by SeriesDescription -> TriggerTime.
* In our case, VENC and velocity direction is read from the SequenceName DICOM HEADER. Code might need to be adjusted if the criteria is different.

## Prediction

To run the prediction, download first the [pre-trained weights](https://auckland.figshare.com/articles/Super_Resolution_4DFlow_MRI/12253424). We have provided an example dataset under the data/ folder.

    1. Create a directory named models/
    2. Put the downloaded 4DFlowNet folder under models/ 
    3. Put your dataset under the data/ folder
    4. Go to src/ and open predictor.py and configure the input_filename and output_filename if necessary
    5. Run predictor.py

Adjustable parameters:

|Param  | Description   | Default|
|------|--------------|--------:|
| patch_size| The image will be split into isotropic patches. Adjust according to computation power and image size.  | 24|
| res_increase| Upsample ratio. Adjustable to any integer. More upsample ratio requires more computation power. *Note*: res_increase=1 will denoise the image at the current resolution |2|
| batch_size| Batch size per prediction. Keep it low. |8|
| round_small_values|Small values are rounded down to zero. Small value is calculated based on venc, according to Velocity per 1 pixel value = venc/2048 |True|
| low_resblock | Number of residual blocks in low resolution space within 4DFlowNet. |8|
| hi_resblock | Number of residual blocks in high resolution space within 4DFlowNet. |4|
    


## Contact Information

If you encounter any problems in using the code, please open an issue in this repository or feel free to contact me by email.

Author: Edward Ferdian (edwardferdian03@gmail.com).
