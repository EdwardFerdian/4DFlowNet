import tensorflow as tf
import numpy as np
import h5py

class PatchHandler3D():
    # constructor
    def __init__(self, data_dir, patch_size, res_increase, batch_size, mask_threshold=0.6):
        self.patch_size = patch_size
        self.res_increase = res_increase
        self.batch_size = batch_size
        self.mask_threshold = mask_threshold

        self.data_directory = data_dir
        self.hr_colnames = ['u','v','w']
        self.lr_colnames = ['u','v','w']
        self.venc_colnames = ['venc_u','venc_v','venc_w']
        self.mag_colnames  = ['mag_u','mag_v','mag_w']
        self.mask_colname  = 'mask'

    def initialize_dataset(self, indexes, shuffle, n_parallel=None):
        '''
            Input pipeline.
            This function accepts a list of filenames with index and patch locations to read.
        '''
        ds = tf.data.Dataset.from_tensor_slices((indexes))
        print("Total dataset:", len(indexes), 'shuffle', shuffle)

        if shuffle:
            # Set a buffer equal to dataset size to ensure randomness
            ds = ds.shuffle(buffer_size=len(indexes)) 

        ds = ds.map(self.load_data_using_patch_index, num_parallel_calls=n_parallel)
        ds = ds.batch(batch_size=self.batch_size)
        
        # prefetch, n=number of items
        ds = ds.prefetch(self.batch_size)
        
        return ds
    
    def load_data_using_patch_index(self, indexes):
        return tf.py_function(func=self.load_patches_from_index_file, 
            # U-LR, HR, MAG, V-LR, HR, MAG, W-LR, HR, MAG, venc, MASK
            inp=[indexes], 
                Tout=[tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32, tf.float32,
                    tf.float32, tf.float32])

    def load_patches_from_index_file(self, indexes):
        # Do typecasting, we need to make sure everything has the correct data type
        # Solution for tf2: https://stackoverflow.com/questions/56122670/how-to-get-string-value-out-of-tf-tensor-which-dtype-is-string
        lr_hd5path = '{}/{}'.format(self.data_directory, bytes.decode(indexes[0].numpy()))
        hd5path    = '{}/{}'.format(self.data_directory, bytes.decode(indexes[1].numpy()))
        
        idx = int(indexes[2])
        x_start, y_start, z_start = int(indexes[3]), int(indexes[4]), int(indexes[5])
        is_rotate = int(indexes[6])
        rotation_plane = int(indexes[7])
        rotation_degree_idx = int(indexes[8])

        patch_size = self.patch_size
        hr_patch_size = self.patch_size * self.res_increase
        
        # ============ get the patch ============ 
        patch_index  = np.index_exp[idx, x_start:x_start+patch_size, y_start:y_start+patch_size, z_start:z_start+patch_size]
        hr_patch_index = np.index_exp[idx, x_start*self.res_increase :x_start*self.res_increase +hr_patch_size ,y_start*self.res_increase :y_start*self.res_increase +hr_patch_size , z_start*self.res_increase :z_start*self.res_increase +hr_patch_size ]
        mask_index = np.index_exp[0, x_start*self.res_increase :x_start*self.res_increase +hr_patch_size ,y_start*self.res_increase :y_start*self.res_increase +hr_patch_size , z_start*self.res_increase :z_start*self.res_increase +hr_patch_size ]
        u_patch, u_hr_patch, mag_u_patch, v_patch, v_hr_patch, mag_v_patch, w_patch, w_hr_patch, mag_w_patch, venc, mask_patch = self.load_vectorfield(hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index)
        
        # ============ apply rotation ============ 
        if is_rotate > 0:
            u_patch, v_patch, w_patch = self.apply_rotation(u_patch, v_patch, w_patch, rotation_degree_idx, rotation_plane, True)
            u_hr_patch, v_hr_patch, w_hr_patch = self.apply_rotation(u_hr_patch, v_hr_patch, w_hr_patch, rotation_degree_idx, rotation_plane, True)
            mag_u_patch, mag_v_patch, mag_w_patch = self.apply_rotation(mag_u_patch, mag_v_patch, mag_w_patch, rotation_degree_idx, rotation_plane, False)
            mask_patch = self.rotate_object(mask_patch, rotation_degree_idx, rotation_plane)
            
        # Expand dims (for InputLayer)
        return u_patch[...,tf.newaxis], v_patch[...,tf.newaxis], w_patch[...,tf.newaxis], \
                    mag_u_patch[...,tf.newaxis], mag_v_patch[...,tf.newaxis], mag_w_patch[...,tf.newaxis], \
                    u_hr_patch[...,tf.newaxis], v_hr_patch[...,tf.newaxis], w_hr_patch[...,tf.newaxis], \
                    venc, mask_patch
                    
    def rotate_object(self, img, rotation_idx, plane_nr):
        if plane_nr==1:
            ax = (0,1)
        elif plane_nr==2:
            ax = (0,2)
        elif plane_nr==3:
            ax = (1,2)
        else:
            # Unspecified rotation plane, return original
            return img

        img = np.rot90(img, k=rotation_idx, axes=ax)
        return img

    def apply_rotation(self, u, v, w, rotation_idx, plane_nr, is_phase_image):
        if rotation_idx == 1:
            # print("90 degrees, plane", plane_nr)
            u,v,w = rotate90(u,v,w, plane_nr, rotation_idx, is_phase_image)
        elif rotation_idx == 2:
            # print("180 degrees, plane", plane_nr)
            u,v,w = rotate180_3d(u,v,w, plane_nr, is_phase_image)
        elif rotation_idx == 3:
            # print("270 degrees, plane", plane_nr)
            u,v,w = rotate90(u,v,w, plane_nr, rotation_idx, is_phase_image)

        return u, v, w

    def load_vectorfield(self, hd5path, lr_hd5path, idx, mask_index, patch_index, hr_patch_index):
        '''
            Load LowRes velocity and magnitude components, and HiRes velocity components
            Also returns the global venc and HiRes mask
        '''
        hires_images = []
        lowres_images = []
        mag_images = []
        vencs = []
        global_venc = 0

        # Load the U, V, W component of HR, LR, and MAG
        with h5py.File(hd5path, 'r') as hl:
            # Open the file once per row, Loop through all the HR column
            for i in range(len(self.hr_colnames)):
                w_hr = hl.get(self.hr_colnames[i])[hr_patch_index]
                # add them to the list
                hires_images.append(w_hr)

            # We only have 1 mask for all the objects in 1 file
            mask = hl.get(self.mask_colname)[mask_index] # Mask value [0 .. 1]
            mask = (mask >= self.mask_threshold) * 1.
            
        with h5py.File(lr_hd5path, 'r') as hl:
            for i in range(len(self.lr_colnames)):
                w = hl.get(self.lr_colnames[i])[patch_index]
                mag_w = hl.get(self.mag_colnames[i])[patch_index]
                w_venc = hl.get(self.venc_colnames[i])[idx]

                # add them to the list
                lowres_images.append(w)
                mag_images.append(mag_w)
                vencs.append(w_venc)
        
        global_venc = np.max(vencs)

        # Convert to numpy array
        hires_images = np.asarray(hires_images)
        lowres_images = np.asarray(lowres_images)
        mag_images = np.asarray(mag_images)
        
        # Normalize the values 
        hires_images = self._normalize(hires_images, global_venc) # Velocity normalized to -1 .. 1
        lowres_images = self._normalize(lowres_images, global_venc)
        mag_images = mag_images / 4095. # Magnitude 0 .. 1

        # U-LR, HR, MAG, V-LR, HR, MAG, w-LR, HR, MAG, venc, MASK
        return lowres_images[0].astype('float32'), hires_images[0].astype('float32'), mag_images[0].astype('float32'), \
            lowres_images[1].astype('float32'), hires_images[1].astype('float32'), mag_images[1].astype('float32'), \
            lowres_images[2].astype('float32'), hires_images[2].astype('float32'), mag_images[2].astype('float32'),\
            global_venc.astype('float32'), mask.astype('float32')
    
    def _normalize(self, u, venc):
        return u / venc

# ---- Rotation code ----
def rotate180_3d(u, v, w, plane=1, is_phase_img=True):
    """
        Rotate 180 degrees to introduce negative values
        xyz Axis stays the same
    """
    if plane==1:
        # Rotate on XY, y*-1, z*-1
        ax = (0,1)
        if is_phase_img:
            v *= -1
            w *= -1
    elif plane==2:
        # Rotate on XZ, x*-1, z*-1
        ax = (0,2)
        if is_phase_img:
            u *= -1
            w *= -1
    elif plane==3:
        # Rotate on YZ, x*-1, y*-1
        ax = (1,2)
        if is_phase_img:
            u *= -1
            v *= -1
    else:
        # Unspecified rotation plane, return original
        return u,v,w
    
    # Do the 180 deg rotation
    u = np.rot90(u, k=2, axes=ax)
    v = np.rot90(v, k=2, axes=ax)
    w = np.rot90(w, k=2, axes=ax)    

    return u,v,w

def rotate90(u, v, w, plane, k, is_phase_img=True):
    """
        Rotate 90 (k=1) or 270 degrees (k=3)
        Introduce axes swapping and negative values
    """
    if plane==1:
        
        ax = (0,1)
        
        if k == 1:
            # =================== ROTATION 90 =================== 
            # Rotate on XY, swap Z to Y +, Y to Z -
            temp = v
            v = w
            w = temp 
            if is_phase_img:
                w *= -1
        elif k == 3:
            # =================== ROTATION 270 =================== 
            # Rotate on XY, swap Z to Y -, Y to Z +
            temp = v
            v = w
            if is_phase_img:
                w *= -1
            w = temp

            

    elif plane==2:
        ax = (0,2)
        if k == 1:
            # =================== ROTATION 90 =================== 
            # Rotate on XZ, swap X to Z +, Z to X -
            temp = w
            w = u
            u = temp 
            if is_phase_img:
                u *= -1
        elif k == 3:
            # =================== ROTATION 270 =================== 
            # Rotate on XZ, swap X to Z -, Z to X +
            temp = w
            w = u
            if is_phase_img:
                w *= -1
            u = temp
        
    elif plane==3:
        ax = (1,2)
        if k == 1:
            # =================== ROTATION 90 =================== 
            # Rotate on YZ, swap X to Y +, Y to X -
            temp = v
            v = u
            u = temp
            if is_phase_img:
                u *= -1
        elif k ==3:
            # =================== ROTATION 270 =================== 
            # Rotate on YZ, swap X to Y -, Y to X +
            temp = v
            v = u
            if is_phase_img:
                v *= -1
            u = temp
    else:
        # Unspecified rotation plane, return original
        return u,v,w
    
    # Do the 90 or 270 deg rotation
    u = np.rot90(u, k=k, axes=ax)
    v = np.rot90(v, k=k, axes=ax)
    w = np.rot90(w, k=k, axes=ax)    

    return u,v,w
