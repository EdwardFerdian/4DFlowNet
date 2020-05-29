from h5functions import save_to_h5
import re

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