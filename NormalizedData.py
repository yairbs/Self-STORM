import random
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import measure
from skimage import io
import torch
from torch.utils.data import Dataset

class NormalizedData(Dataset):
    def __init__(self, root, is_normalized=False, dest=None, mask_threshold=0.95, label_threshold=150, modified_db_size='original'):
        super(NormalizedData, self).__init__()
        self.dataset = io.imread(root)
        self.save_path = dest
        self.img_size = self.dataset.shape[1]
        self.frame_count = self.dataset.shape[0]
        self.mask_threshold = mask_threshold
        self.label_threshold = label_threshold
        self.modified_db_size = modified_db_size

        # Expanding the shape from NxWxH to NxCxWxH where C is the number of channels which for this project it is
        # currently 1 only for grayscale.
        non_normalized_data = np.expand_dims(self.dataset, 1)
        self.data = non_normalized_data

        if not is_normalized:
            self.data = self.create_modified_dataset()
        else:
            X = np.array(self.dataset).astype(float)
            self.data = torch.tensor(X).unsqueeze(1)

    def positive_normalize_img(self, img):
        normalized_img =  np.nan_to_num(
            (img - np.mean(img)) /
            np.std(img)
        )
        return normalized_img-np.min(normalized_img)

    def get_normalized_mean_img(self):
        precentile = np.quantile(self.data.reshape((self.frame_count,-1)),self.mask_threshold,axis=1,keepdims=True).reshape(-1,1,1,1)
        data_masked = np.where(self.data>precentile,1,0)
        mean_img = np.sum(data_masked*self.data, 0)/self.data.shape[0]
        min_val = np.min(mean_img)
        moved_img = mean_img-min_val
        max_val = np.max(moved_img)
        return moved_img/max_val
    
    def label_phlorophores(self,data):
        blur_radius = 1
        precentile = np.quantile(data.reshape((self.frame_count, -1)), self.mask_threshold, axis=1, keepdims=True).reshape(-1, 1, 1, 1)
        data_masked = np.where(data > precentile, 1, 0)
        imgf = ndimage.gaussian_filter(data_masked * 255, blur_radius)

        labeled, nr_objects = ndimage.label(imgf > self.label_threshold, structure =
        [[[[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],
         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]
         ],
        [[[0,0,0],
         [0,0,0],
         [0,0,0]],
         [[0,1,0],
         [1,1,1],
         [0,1,0]],
         [[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]
         ],
         [[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]],
          [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]],
          [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]]
          ]])
        
        return labeled
    
    def find_bounding_box(self,labeled_mask):
        return measure.regionprops(labeled_mask.reshape(self.img_size,self.img_size))
    
    def find_centroid_in_region(self,label_img,img):
        mask = label_img != 0
        if not np.any(mask):
            return 0
        return np.unravel_index(np.argmax(mask*img),mask.shape)
    
    def set_flourophore_dict(self,labeled_masks,data,disp=True):
        regions = []
        flourophore_dict = {}

        for i, label_img in enumerate(labeled_masks):
            regions.append(self.find_bounding_box(label_img))

            for props in regions[i]:
                minr, minc, maxr, maxc = props.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                boxed_label_img = label_img[0,minr:maxr,minc:maxc]
                boxed_input_img = data[i,0,minr:maxr,minc:maxc]

                centroid = self.find_centroid_in_region(boxed_label_img,boxed_input_img)
                if centroid == 0:
                    continue
                centroid_x = int(centroid[1]+minc)
                centroid_y = int(centroid[0]+minr)
                if not centroid_x in flourophore_dict:
                    flourophore_dict[centroid_x] = {}
                if not centroid_y in flourophore_dict[centroid_x]:
                    flourophore_dict[centroid_x][centroid_y] = []
                normalized_box_image = self.positive_normalize_img(boxed_input_img)
                if normalized_box_image.sum() > 0:
                    flourophore_dict[centroid_x][centroid_y].append((i,boxed_input_img,normalized_box_image,minc,minr))
                if disp:
                    plt.plot(bx, by, '-r', linewidth=0.2, alpha=0.05)
                    plt.plot(centroid[1]+minc, centroid[0]+minr, marker="o", markersize=0.3, alpha=0.5)
        
        return flourophore_dict
    
    def build_dataset_from_dict(self, flourophore_dict, db_size=None):
        flourophore_options = []
        # create all the options of x,y combinations to sample from
        accum_len = 0
        for x_key in flourophore_dict:
            for y_key in flourophore_dict[x_key]:
                    flourophore_options.append((x_key,y_key))
                    accum_len += len(flourophore_dict[x_key][y_key])

        if db_size == 'original':
            # Sample according to original temporal occurance of each emitter
            new_db = np.zeros((self.frame_count,1,self.img_size,self.img_size))
            for img_num in range(self.frame_count):
                cur_img = np.zeros((self.img_size,self.img_size))
                for x_key in flourophore_dict:
                    for y_key in flourophore_dict[x_key]:
                        temp_img = np.zeros((self.img_size,self.img_size))
                        count = 0
                        for d in flourophore_dict[x_key][y_key]:
                            (t_idx, boxed_img, boxed_norm_img, box_x, box_y) = d
                            (y_size, x_size) = boxed_img.shape
                            if t_idx == img_num:
                                temp_img[box_y:box_y + y_size, box_x:box_x + x_size] += boxed_norm_img
                                count+=1
                        cur_img += temp_img
                new_db[img_num,:,:,:] = cur_img

        elif db_size == 'average':
            # Average of options per location
            new_db = np.zeros((len(flourophore_options),1,self.img_size,self.img_size))
            for img_num in range(len(flourophore_options)):
                (x_key,y_key) = flourophore_options[img_num]
                for d in flourophore_dict[x_key][y_key]:
                    temp_img = np.zeros((self.img_size,self.img_size))
                    (boxed_img, boxed_norm_img, box_x, box_y) = d
                    (y_size, x_size) = boxed_img.shape
                    temp_img[box_y:box_y + y_size, box_x:box_x + x_size] += boxed_img
                temp_img = self.positive_normalize_img(temp_img)
                new_db[img_num,:,:,:] = temp_img

        elif db_size == 'single_frame':
            # Exact sampling of each option once
            new_db = np.zeros((accum_len,1,self.img_size,self.img_size))
            img_idx = 0
            for img_num in range(len(flourophore_options)):
                (x_key,y_key) = flourophore_options[img_num]
                for d in flourophore_dict[x_key][y_key]:
                    temp_img = np.zeros((self.img_size,self.img_size))
                    (boxed_img, boxed_norm_img, box_x, box_y) = d
                    (y_size, x_size) = boxed_img.shape
                    temp_img[box_y:box_y + y_size, box_x:box_x + x_size] += boxed_norm_img
                    new_db[img_idx,:,:,:] = temp_img
                    img_idx += 1

        elif db_size:
            # Random sampling of the options (according to db_size)
            x = len(flourophore_options)//db_size
            max_flourophores_per_image = x if x >= 1 else 1
            new_db = np.zeros((db_size,1,self.img_size,self.img_size))
            for img_num in range(db_size):
                temp_img = np.zeros((self.img_size,self.img_size))
                num_of_flourophores = random.randint(1,max_flourophores_per_image)
                for i in range(num_of_flourophores):
                    (x_key,y_key) = random.choice(flourophore_options)
                    (boxed_img, boxed_norm_img, box_x, box_y) = random.choice(flourophore_dict[x_key][y_key])
                    (y_size, x_size) = boxed_img.shape
                    temp_img[box_y:box_y + y_size, box_x:box_x + x_size] += boxed_norm_img
                new_db[img_num,:,:,:] = temp_img

        else:
            print("Invalid resampling mode!")

        return new_db

    def create_modified_dataset(self):
        labeled_mask = self.label_phlorophores(self.data)
        self.flourophore_dict = self.set_flourophore_dict(labeled_mask,self.data,False)
        self.modified_data = self.build_dataset_from_dict(self.flourophore_dict,self.modified_db_size)
        self.frame_count = self.modified_db_size
        self.modified_data = self.modified_data.reshape(self.modified_data.shape[0],self.img_size,self.img_size)
        if self.save_path is not None:
            io.imsave(self.save_path,self.modified_data)
        X = np.array(self.modified_data).astype(float)
        data = torch.tensor(X).unsqueeze(1)
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx]
        return image