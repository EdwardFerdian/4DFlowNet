import h5py
import numpy as np

class ImageDataset():
    def __init__(self):
        self.velocity_colnames   = ['u', 'v', 'w']
        self.venc_colnames = ['venc_u', 'venc_v', 'venc_w']
        self.mag_colnames  = ['mag_u', 'mag_v', 'mag_w']
        self.dx_colname = 'dx'

    def _set_images(self, velocity_images, mag_images, venc, dx):
        '''
            Called by load_vectorfield
        '''
        # Normalize the values first
        velocity_images = self._normalize(velocity_images, venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1

        # Set the attributes
        self.u = velocity_images[0].astype('float32')
        self.v = velocity_images[1].astype('float32')
        self.w = velocity_images[2].astype('float32')
        
        self.mag_u = mag_images[0].astype('float32')
        self.mag_v = mag_images[1].astype('float32')
        self.mag_w = mag_images[2].astype('float32')

        # Keep the venc to denormalized data
        self.venc = venc.astype('float32')
        # Calculate PX sensitivity to zero out the predictions later
        self.velocity_per_px = self.venc / 2048
        self.dx = dx

    def _normalize(self, velocity, venc):
        return velocity / venc

    def postprocess_result(self, results, zerofy=True):
        # Denormalized the data
        results = results * self.venc 
        
        if zerofy:
            print(f"Zero out velocity component less than {self.velocity_per_px}")
            # remove small velocity values
            results[np.abs(results) < self.velocity_per_px] = 0
        return results

    def get_dataset_len(self, filepath):
        with h5py.File(filepath, 'r') as hl:
            data_size = hl[self.velocity_colnames[0]].shape[0]
        return data_size
   
    def load_vectorfield(self, filepath, idx):
        '''
            Override the load u v w data by adding some padding in xy planes
        '''
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0
        dx = None

        # Load the U, V, W component of LR, and MAG
        with h5py.File(filepath, 'r') as hl:
            if self.dx_colname in hl:
                dx = hl.get(self.dx_colname)[idx]
            
            for i in range(len(self.velocity_colnames)):                
                w = np.asarray(hl.get(self.velocity_colnames[i])[idx])
                mag_w = np.asarray(hl.get(self.mag_colnames[i])[idx])
                w_venc = np.asarray(hl.get(self.venc_colnames[i])[idx])
            
                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)

        # /end of u,v,w loop
        global_venc = np.max(vencs)
        
        # Convert to numpy array
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)

        # Setup the class properties
        self._set_images(lowres_images, mag_images, global_venc, dx)
    