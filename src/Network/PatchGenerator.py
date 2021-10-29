import numpy as np
import h5py
from utils import ImageDataset

class PatchGenerator():
    def __init__(self, patch_size, res_increase):
        self.patch_size = patch_size
        self.effective_patch_size = patch_size - 4 # we strip down 2 from each sides (on LR)
        self.res_increase = res_increase
        # we make sure we pad it on the far side of x,y,z so the division will match
        self.padding = (0,0,0) 

    def patchify(self, dataset: ImageDataset):
        """
            Create overlapping patch of size of patch_size
            On LR, we exclude 2 px from each side, effectively the size being used is patch_size-4
            On HR, the excluded pixels are (2*res_increase) from each side
        """
        u_stacks, i,j,k = self._generate_overlapping_patches(dataset.u)
        v_stacks, i,j,k = self._generate_overlapping_patches(dataset.v)
        w_stacks, i,j,k = self._generate_overlapping_patches(dataset.w)
        umag_stacks, i,j,k = self._generate_overlapping_patches(dataset.mag_u)
        vmag_stacks, i,j,k = self._generate_overlapping_patches(dataset.mag_v)
        wmag_stacks, i,j,k = self._generate_overlapping_patches(dataset.mag_w)
        
        # Store this info for unpatchify
        self.nr_x = i
        self.nr_y = j
        self.nr_z = k
        
        # Expand dims for tf.keras input shape
        u_stacks = np.expand_dims(u_stacks, -1)
        v_stacks = np.expand_dims(v_stacks, -1)
        w_stacks = np.expand_dims(w_stacks, -1)

        umag_stacks = np.expand_dims(umag_stacks, -1)
        vmag_stacks = np.expand_dims(vmag_stacks, -1)
        wmag_stacks = np.expand_dims(wmag_stacks, -1)

        return (u_stacks, v_stacks, w_stacks), (umag_stacks, vmag_stacks, wmag_stacks)
    
    def unpatchify(self, results):
        """
            Reconstruct the 3-velocity components back to its original shape
        """
        prediction_u = self._patchup_with_overlap(results[:,:,:,:,0], self.nr_x, self.nr_y, self.nr_z)
        prediction_v = self._patchup_with_overlap(results[:,:,:,:,1], self.nr_x, self.nr_y, self.nr_z)
        prediction_w = self._patchup_with_overlap(results[:,:,:,:,2], self.nr_x, self.nr_y, self.nr_z)

        #return predictions
        return prediction_u, prediction_v, prediction_w

    def _pad_to_patch_size_with_overlap(self, img):
        """
            Pad image to the right, until it is exactly divisible by patch size
        """
        side_pad = (self.patch_size-self.effective_patch_size) // 2
        
        # mandatory padding
        img = np.pad(img, ((side_pad, side_pad),(side_pad, side_pad),(side_pad, side_pad)), 'constant')
        
        res_x = (img.shape[0] % self.effective_patch_size)
        if (res_x > (2* side_pad)):
            pad_x = self.patch_size - res_x
        else:
            pad_x = (2 * side_pad) - res_x

        res_y = (img.shape[1] % self.effective_patch_size)
        if (res_y > (2* side_pad)):
            pad_y = self.patch_size - res_y
        else:
            pad_y = (2 * side_pad) - res_y

        res_z = (img.shape[2] % self.effective_patch_size)
        if (res_z > (2* side_pad)):
            pad_z = self.patch_size - res_z
        else:
            pad_z = (2 * side_pad) - res_z
        
        img = np.pad(img, ((0, pad_x),(0, pad_y),(0, pad_z)), 'constant')

        # the padding is for the HiRes version because we need to reconstruct the result later
        self.padding = (pad_x*self.res_increase, pad_y*self.res_increase, pad_z*self.res_increase)
        # print("LR padding:", self.padding)
        
        return img

    def _generate_overlapping_patches(self, img):
        patch_size = self.patch_size
        
        img = self._pad_to_patch_size_with_overlap(img)

        all_pads = (self.patch_size - self.effective_patch_size)

        u_stack = []
        
        nr_x = (img.shape[0]-all_pads) // self.effective_patch_size
        nr_y = (img.shape[1]-all_pads) // self.effective_patch_size
        nr_z = (img.shape[2]-all_pads) // self.effective_patch_size
        
        for i in range(nr_x):
            x_start = i * self.effective_patch_size #stride x
            for j in range(nr_y):
                y_start = j * self.effective_patch_size #stride y
                for k in range(nr_z):
                    z_start = k * self.effective_patch_size #stride z
                    
                    patch_index  = np.index_exp[x_start:x_start+patch_size, y_start:y_start+patch_size, z_start:z_start+patch_size]
                    
                    u_loop = img[patch_index]
                    u_stack.append(u_loop)
                    # print(patch_index)              
        return np.asarray(u_stack), nr_x, nr_y, nr_z
            # return the number of of i j k elements        

    def _patchup_with_overlap(self, patches, x, y, z):
        # print("Prediction size:", patches.shape)

        side_pad = (self.patch_size - self.effective_patch_size) // 2
        side_pad_hr =  side_pad * self.res_increase
        patch_size = patches.shape[1]
        n = patch_size-side_pad_hr

        patches = patches[:,side_pad_hr:n, side_pad_hr:n, side_pad_hr:n]
        # patches = patches[:,side_pad_hr:-side_pad_hr, side_pad_hr:-side_pad_hr, side_pad_hr:-side_pad_hr]
        
        z_stacks = []
        for k in range(len(patches) // z):
            
            z_start =k*z
            # print('z_start', z_start, k, z)
            z_stack = np.concatenate(patches[z_start:z_start+z], axis=2)
            # print('z_stack', z_stack.shape)
            z_stacks.append(z_stack)

        y_stacks = []
        for j in range(len(z_stacks) // y):
            y_start =j*y
            # print('y_start', y_start, j, y)
            y_stack = np.concatenate(z_stacks[y_start:y_start+y], axis=1)
            # print('y_stack', y_stack.shape)
            y_stacks.append(y_stack)

        end_results = np.concatenate(y_stacks, axis=0)

        # crop based on the padding we did during patchify
        if self.padding[0] > 0:
            end_results = end_results[:-self.padding[0],:, :]
        if self.padding[1] > 0:
            end_results = end_results[:, :-self.padding[1],:]
        if self.padding[2] > 0:
            end_results = end_results[:, :, :-self.padding[2]]

        return end_results     
    
    