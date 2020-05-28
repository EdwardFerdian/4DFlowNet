import random as rnd
import numpy as np
import csv

def write_header(filename):
    fieldnames = ['source', 'target','index', 'start_x', 'start_y', 'start_z', 'rotate', 'rotation_plane', 'rotation_degree_idx', 'coverage']
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def generate_random_patches(input_filename, target_filename, output_filename, index, n_patch, binary_mask, patch_size, minimum_coverage, empty_patch_allowed, apply_all_rotation=True):
    empty_patch_counter = 0
            
    # foreach row, create n number of patches
    j = 0
    not_found = 0
    while j < n_patch:
        if not_found > 100:
            print(f"Cannot find enough patches above {minimum_coverage} coverage, please lower the minimum_coverage")
            break

        can_still_take_empty_patch = empty_patch_counter < empty_patch_allowed
        # print('patch number', j)
        patch = PatchData(input_filename, target_filename, patch_size)
        
        # default, no rotation
        patch.create_random_patch(binary_mask, index)
        patch.calculate_patch_coverage(binary_mask, minimum_coverage)

        # before saving, do a check
        # print(patch.coverage)
        if patch.coverage < minimum_coverage:
            if can_still_take_empty_patch:
                
                print('Taking this empty one',patch.coverage)
                empty_patch_counter += 1
                
            else:
                # skip empty patch because we already have one
                not_found += 1
                continue

        patch.write_to_csv(output_filename)

        # apply ALL  rotation
        if apply_all_rotation:
            patch.rotate = 1
            for plane_nr in range(1,4):
                # rotation plane 1,2,3
                patch.rotation_plane = plane_nr

                for rotation_idx in range(1,4):
                    # rotation index 1,2,3
                    patch.rotation_degree_idx = rotation_idx
                    patch.write_to_csv(output_filename)
                # /end of rotation idx
            # /end of plane
        else:
            patch.rotate = 1
            # do 1 random rotation
            patch.rotation_plane = rnd.randint(1,3)
            patch.rotation_degree_idx = rnd.randint(1,3)
            patch.write_to_csv(output_filename)

        # break
        j += 1
    # /end of while n_patch
    
class PatchData:
    def __init__(self, source_file, target_file, patch_size):
        self.patch_size = patch_size

        self.source_file = source_file
        self.target_file = target_file
        self.idx = None
        self.start_x = None
        self.start_y = None
        self.start_z = None
        self.rotate = 0
        self.rotation_plane = 0
        self.rotation_degree_idx = 0
        self.coverage = 0

    def create_random_patch(self, u, index):
        self.idx = index
        self.start_x = rnd.randrange(0, u.shape[0] - self.patch_size + 1) 
        self.start_y = rnd.randrange(0, u.shape[1] - self.patch_size + 1) 
        self.start_z = rnd.randrange(0, u.shape[2] - self.patch_size + 1) 

    def set_patch(self, index, x, y, z):
        self.idx = index
        self.start_x = x
        self.start_y = y
        self.start_z = z

    def calculate_patch_coverage(self, binary_mask, minimum_coverage=0.2):
        patch_region = np.index_exp[self.start_x:self.start_x+self.patch_size, self.start_y:self.start_y+self.patch_size, self.start_z:self.start_z+self.patch_size]
        patch = binary_mask[patch_region]

        self.coverage = np.count_nonzero(patch) / self.patch_size ** 3
        self.coverage = np.round(self.coverage * 1000) / 1000 # round the number to 3 decimal digit

    def create_random_rotation(self):
        is_rotate = rnd.randint(0,1) # 50-50 percent chance of augmentation
        if is_rotate == 0:
            plane_nr = 0
            degree_idx = 0
        else:
            plane_nr = rnd.randint(1,3)
            degree_idx = rnd.randint(1,3)
        
        self.rotate = is_rotate
        self.rotation_plane = plane_nr
        self.rotation_degree_idx = degree_idx

        return is_rotate, plane_nr, degree_idx

    def write_to_csv(self, output_filename):
        fieldnames = ['source', 'target', 'index', 'start_x', 'start_y', 'start_z', 'rotate', 'rotation_plane', 'rotation_degree_idx', 'coverage']
        with open(output_filename, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'source': self.source_file, 'target': self.target_file, 'index': self.idx, 
            'start_x': self.start_x, 'start_y': self.start_y, 'start_z': self.start_z,
            'rotate': self.rotate, 'rotation_plane': self.rotation_plane, 'rotation_degree_idx': self.rotation_degree_idx,
            'coverage': self.coverage})