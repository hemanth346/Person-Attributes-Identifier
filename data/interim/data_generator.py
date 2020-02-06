# Class to generate batches of image data to be fed to model

import os
import os.path as path
import numpy as np
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array

class PersonDataGenerator(keras.utils.Sequence):
    """
    Ground truth data generator 

    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """
    
    def __init__(self, df, batch_size = 32, input_size = (224, 224), 
                 location = '.', augmentations = None, save_dir = None,
                 shuffle = False):
        self.df = df
        self.image_size = input_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentations #ImageDataGenerator instance
        self.location = location
        self.save_dir =  save_dir # path.abspath(path.join(self.location,'processed'))
        self.on_epoch_end()

        if self.save_dir:
            self.save_dir = path.abspath(self.save_dir)
            if not path.isdir(self.save_dir):
                os.mkdirs(self.save_dir, exist_ok=True)

        # Label columns per attribute
        self._gender_cols_ = [col for col in df.columns if col.startswith("gender")]
        self._imagequality_cols_ = [col for col in df.columns if col.startswith("imagequality")]
        self._age_cols_ = [col for col in df.columns if col.startswith("age")]
        self._weight_cols_ = [col for col in df.columns if col.startswith("weight")]
        self._carryingbag_cols_ = [col for col in df.columns if col.startswith("carryingbag")]
        self._footwear_cols_ = [col for col in df.columns if col.startswith("footwear")]
        self._emotion_cols_ = [col for col in df.columns if col.startswith("emotion")]
        self._bodypose_cols_ = [col for col in df.columns if col.startswith("bodypose")]

    def __len__(self):
        """
        Number of batch in the Sequence.
        """
        return int(np.floor(self.df.shape[0] / self.batch_size))

    
    def __getitem__(self, index):
        """
        Gets batch at position index.
        fetch batched images and targets        
        """
        # slice function - https://www.w3schools.com/python/ref_func_slice.asp
        
        
        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        items = self.df.iloc[batch_slice]

        images = np.stack([cv2.imread(path.join(self.location, item["image_path"])) for _, item in items.iterrows()])        
#         if self.augmentation:
#             if self.save_dir:
#                 images = self.augmentation.flow(images, 
#                                             batch_size=self.batch_size, 
#                                             save_to_dir=self.save_dir,
#                                             save_prefix='aug').next()
                    
#             else:
        images = self.augmentation.flow(images, 
                                            batch_size=self.batch_size).next()


        target = {
            "gender_output": items[self._gender_cols_].values,
            "image_quality_output": items[self._imagequality_cols_].values,
            "age_output": items[self._age_cols_].values,
            "weight_output": items[self._weight_cols_].values,
            "bag_output": items[self._carryingbag_cols_].values,
            "pose_output": items[self._bodypose_cols_].values,
            "footwear_output": items[self._footwear_cols_].values,
            "emotion_output": items[self._emotion_cols_].values,
        }
        
        return images, target

    def on_epoch_end(self):
        """
        Shuffles/sample the df and thereby 
        updates indexes after each epoch
        
        Method called at the end of every epoch.
        
        """
        if self.shuffle == True:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            # frac --> take sample of the given df, sample size is given as fraction number
            # reset_index drop --> use the drop parameter to avoid the old index being added as a column
            # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
            