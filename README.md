# 4DFlowNet
Super Resolution 4D Flow MRI using Residual Neural Network

This is an implementation of the paper [4DFlowNet: Super-Resolution 4D Flow MRI Using Deep Learning and Computational Fluid Dynamics](https://www.frontiersin.org/articles/10.3389/fphy.2020.00138/full) using Tensorflow 1.8.0. 

Please find the pre-trained networks weights here:

[4DFlowNet pre-trained weights](https://auckland.figshare.com/articles/Super_Resolution_4DFlow_MRI/12253424)

[4DFlowNet pre-trained weights (temporary link)](https://bit.ly/2zgs3GX)


If you are using later Tensorflow 1.x version that is not compatible with this version, please refer to Tensorflow backwards compatibility (tf.compat module). 

We are transitioning to Tensorflow 2.0. Stay tuned for an updated version.

## Prepare data

## Training

(in progress)

## Prediction

To run the prediction, download first the pre-trained weights. We have provided an example dataset under the data/ folder.

    1. Create a directory named models/
    2. Put the 4DFlowNet folder under models/ 
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
    

# Example results

Below are the example results from an actual 4D Flow MRI of a bifurcation phantom dataset. 


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

## Contact Information

If you encounter any problems in using the code, please open an issue in this repository.

Author: Edward Ferdian (edwardferdian03@gmail.com).
