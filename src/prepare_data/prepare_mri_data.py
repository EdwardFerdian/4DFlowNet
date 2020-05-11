
import numpy as np
import pydicom
import h5py
import os
import re
import argparse

class DicomData:
    def __init__(self):
        self.sequenceNames = []
        self.spacing = []

        self._phaseImages = []
        self._magImages = []

        # vel and mag Components
        self.u = None
        self.v = None
        self.w = None

        self.u_mag = None
        self.v_mag = None
        self.w_mag = None

        self.u_venc = None
        self.v_venc = None
        self.w_venc = None
    
    def print(self):
        attributes = vars(self)
        for item in attributes:
            if not item.startswith('_'):
                print (item , ' : ' , attributes[item])

    def phase_to_velocity(self, phase_image, venc):
        """
            Phase image range: 0-4096, with 2048 as 0 velocity (in m/s)
        """
        return (phase_image - 2048.) / 2048. * venc / 100.

    def determine_velocity_components(self, in_multiplier, fh_multiplier, rl_multiplier):
        """ 
            Determine the velocity direction and venc from the sequence name 
        """
        # print("Calculating velocity components...")
        for i in range(len(self._phaseImages)):
            seqName = self.sequenceNames[i]
            phase_image = self._phaseImages[i]
            mag_image = self._magImages[i]

            # Check venc from the sequence name (e.g. fl3d1_v150fh)
            pattern = re.compile(".*?_v(\\d+)(\\w+)")
            found = pattern.search(seqName)
            
            assert found, "Venc pattern not found, please check your DICOM header."
        
            venc = int(found.group(1))
            direction = found.group(2)
            # print('venc_direction', direction, venc)

            # Convert the phase image to velocity
            phase_image = self.phase_to_velocity(phase_image, venc)

            # Note: This is based on our DICOM header. The direction is a bit confusing.
            # In our case we always have in/ap/fh combination
            if direction == "in":
                self.u      = phase_image * in_multiplier
                self.u_mag  = mag_image
                self.u_venc = venc/100
            elif direction == "rl" or direction == "ap":
                self.w      = phase_image * rl_multiplier
                self.w_mag  = mag_image
                self.w_venc = venc/100
            else: # "fh" 
                self.v      = phase_image * fh_multiplier
                self.v_mag  = mag_image
                self.v_venc = venc/100

    def save_dataset(self, output_filepath, triggerTime):
        assert self.u is not None, "Please calculate velocity components first"

        save_to_h5(output_filepath, "triggerTimes", float(triggerTime))
       
        save_to_h5(output_filepath, "u", self.u)
        save_to_h5(output_filepath, "v", self.v)
        save_to_h5(output_filepath, "w", self.w)

        save_to_h5(output_filepath, "mag_u", self.u_mag)
        save_to_h5(output_filepath, "mag_v", self.v_mag)
        save_to_h5(output_filepath, "mag_w", self.w_mag)

        save_to_h5(output_filepath, "venc_u", self.u_venc)
        save_to_h5(output_filepath, "venc_v", self.v_venc)
        save_to_h5(output_filepath, "venc_w", self.w_venc)

        save_to_h5(output_filepath, "dx", self.spacing)

def save_to_h5(output_filepath, col_name, dataset):
    dataset = np.expand_dims(dataset, axis=0)

    # convert float64 to float32 to save space
    if dataset.dtype == 'float64':
        dataset = np.array(dataset, dtype='float32')
    
    with h5py.File(output_filepath, 'a') as hf:    
        if col_name not in hf:
            datashape = (None, )
            if (dataset.ndim > 1):
                datashape = (None, ) + dataset.shape[1:]
            hf.create_dataset(col_name, data=dataset, maxshape=datashape)
        else:
            hf[col_name].resize((hf[col_name].shape[0]) + dataset.shape[0], axis = 0)
            hf[col_name][-dataset.shape[0]:] = dataset


def get_filepaths(directory):
    """
        This function will generate the file names in a directory 
        tree by walking the tree either top-down or bottom-up. For each 
        directory in the tree rooted at directory top (including top itself), 
        it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths 

def get_volume(vol_dir):
    """
        Get phase or magnitude image volume
    """
    volume = []
    # Retrieve all the dicom filepaths
    files  = get_filepaths(vol_dir)
    
    for slice_nr, dicom_path in enumerate(files):
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array
        
        if slice_nr == 0:
            # Get this on the first slice only
            spacing = ds.PixelSpacing
            spacing.append(ds.SliceThickness)
            spacing = np.asarray(spacing)
            
            # Note: In our case, sequence name contains venc and direction info
            sequence_name = ds.SequenceName
            # print(sequence_name)

        volume.append(img)
    volume = np.asarray(volume)
    return volume, spacing, sequence_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory contains the Dicom files with [Phase/Magnitude series desc]/[triggerTime] structure')
    parser.add_argument('--output-dir', type=str, default='../Dataset', help='Output directory where the dataset will be saved')
    parser.add_argument('--output-filename', type=str, default='mri_data.h5', help='Output filename (HDF5)')
    parser.add_argument('--phase-pattern', type=str, default='_P_', help='Pattern of phase series description')
    parser.add_argument('--mag-pattern', type=str, default='_M_', help='Pattern of magnitude series description')
    parser.add_argument('--fh-mul', type=int, default= -1, help='Velocity multiplier for Foot-Head direction')
    parser.add_argument('--rl-mul', type=int, default=  1, help='Velocity multiplier for Right-Left direction')
    parser.add_argument('--in-mul', type=int, default=  1, help='Velocity multiplier for Inplane direction')
    args = parser.parse_args()
    
    case_dir      = args.input_dir
    phase_pattern = args.phase_pattern
    mag_pattern   = args.mag_pattern
    output_path   = args.output_dir
    output_filename = args.output_filename
    in_multiplier = args.in_mul
    fh_multiplier = args.fh_mul
    rl_multiplier = args.rl_mul
    
    output_filepath = f'{output_path}/{output_filename}'

    # 1. Get the phase and magnitude directories 
    directories = os.listdir(case_dir)
    phase_dirs = [item for item in directories if phase_pattern in item]
    mag_dirs   = [item for item in directories if mag_pattern   in item]
    print("Phase dirs:\n", "\n".join(phase_dirs))
    print("Mag dirs:\n"  , "\n".join(mag_dirs))

    assert len(phase_dirs) == 3, f"There must be exactly 3 Phase directories with {phase_pattern} pattern"
    assert len(mag_dirs)   == 3, f"There must be exactly 3 Magnitude directories with {mag_pattern} pattern"

    # 2. Get and sort the trigger times
    dirpath = f'{case_dir}/{phase_dirs[0]}'
    timeFrames = os.listdir(dirpath)
    timeFrames = sorted(timeFrames, key=float)
    # print('All frames sorted:', timeFrames)

    # Create the output dir
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 3. Looping through the time frames
    for j in range(0, len(timeFrames)):
        triggerTime = timeFrames[j]
        print(f"\rProcessing {j+1}/{len(timeFrames)} (frame {triggerTime})          ", end="\r")

        # Wrap it as DicomData instance
        dicom_data = DicomData()
        # Collect the 3 phase and magnitude volumes for 1 time frame
        for mag_dir, p_dir in zip(mag_dirs, phase_dirs):
            magnitude_path = f'{case_dir}/{mag_dir}/{triggerTime}'
            phase_path = f'{case_dir}/{p_dir}/{triggerTime}'

            # Get the magnitude and phase images
            mag_images, _, _                = get_volume(magnitude_path)
            phase_images, spacing, sequence = get_volume(phase_path)
            
            dicom_data._phaseImages.append(phase_images)
            dicom_data._magImages.append(mag_images)
            dicom_data.sequenceNames.append(sequence)
            dicom_data.spacing = spacing
        # Save per row
        dicom_data.determine_velocity_components(in_multiplier, fh_multiplier, rl_multiplier)
        dicom_data.save_dataset(output_filepath, triggerTime)

    # End of trigger time loop
    print(f'\nDone! saved at {output_filepath}')


if __name__ == '__main__':
    main()