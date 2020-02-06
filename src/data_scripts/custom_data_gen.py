# Class to generate batches of image data to be fed to model
# inclusive of both original data and augmented data


# example 
        # train_gen = PersonDataGenerator(
        #     train_df, 
        #     batch_size=32, 
        #     aug_list=[
        #         ImageDataGenerator(rotation_range=45),
        #         ImageDataGenerator(horizontal_flip=True),
        #         ImageDataGenerator(vertical_flip=True),
        #     ],
        #     incl_orig=True,  # Whether to include original images
        # )

from __future__ import division

import os
import os.path as path
import numpy as np
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array


class PersonDataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size=32, shuffle=True, aug_list=[], incl_orig=True):
        """        Ground truth data batch generator    """
        self.df = df
        self.batch_size=batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
        self.aug_list = aug_list
        self.incl_orig = incl_orig
        self.orig_len = int(np.floor(self.df.shape[0] / self.batch_size))


        # Label columns per attribute
        self._gender_cols_          = [col for col in df.columns if col.startswith("gender")]
        self._imagequality_cols_    = [col for col in df.columns if col.startswith("imagequality")]
        self._age_cols_             = [col for col in df.columns if col.startswith("age")]
        self._weight_cols_          = [col for col in df.columns if col.startswith("weight")]
        self._carryingbag_cols_     = [col for col in df.columns if col.startswith("carryingbag")]
        self._footwear_cols_        = [col for col in df.columns if col.startswith("footwear")]
        self._emotion_cols_         = [col for col in df.columns if col.startswith("emotion")]
        self._bodypose_cols_        = [col for col in df.columns if col.startswith("bodypose")]


    def __len__(self):
        """
        Number of batches in the Sequence(i.e per epoch).
        """

        if self.incl_orig:
            delta = 1
        else:
            delta = 0
        return self.orig_len * (len(self.aug_list) + delta)

    def __getitem__(self, index):
        """
        Gets batches of images - generates sets of images 
        based on augementation strategies, can include 
        original images as well - Original images will be 
        rescaled while generating batches

        fetch batches of image data and targets        
        """
        if not self.incl_orig :
            index += self.orig_len - 1

        if index > self.orig_len - 1:
            aug = self.aug_list[index // self.orig_len - 1]
            index %= self.orig_len
        else:
            aug = None

        batch_slice = slice(index * self.batch_size, (index + 1) * self.batch_size)
        items = self.df.iloc[batch_slice]

        images = np.stack([(cv2.imread(item["image_path"])/255.0) for _, item in items.iterrows()])
        
        if aug is not None:
            images = aug.flow(images, batch_size=self.batch_size, shuffle=False).next()
        
        target = {
            "gender_output"         : items[self._gender_cols_].values,
            "image_quality_output"  : items[self._imagequality_cols_].values,
            "age_output"            : items[self._age_cols_].values,
            "weight_output"         : items[self._weight_cols_].values,
            "bag_output"            : items[self._carryingbag_cols_].values,
            "pose_output"           : items[self._bodypose_cols_].values,
            "footwear_output"       : items[self._footwear_cols_].values,
            "emotion_output"        : items[self._emotion_cols_].values,
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


