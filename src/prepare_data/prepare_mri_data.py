
import numpy as np
import pydicom
import os
import argparse
from DicomData import DicomData

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
    parser.add_argument('--output-dir', type=str, default='../../Dataset', help='Output directory where the dataset will be saved')
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