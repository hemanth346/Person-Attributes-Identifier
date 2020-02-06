### Objective :
Given an image of a person, predict eight different attributes, each with multiple values, using the image.

##### This is a Multi-class Multi-Label problem

### Dataset
Images have been annotated and labelled by our group who have been working on the problem individually.

We have a total of 13573 images labelled with human bias.

All images have been cropped, centered and resized to 224x224px

###### Attributes and corresponding labels
```
gender     	 :  ['male' 'female']
imagequality    :  ['Average' 'Good' 'Bad']
age        	 :  ['35-45' '45-55' '25-35' '15-25' '55+']
weight     	 :  ['normal-healthy' 'over-weight' 'slightly-overweight' 'underweight']
carryingbag 	:  ['Grocery/Home/Plastic Bag' 'None' 'Daily/Office/Work Bag']
footwear   	 :  ['Normal' 'CantSee' 'Fancy']
emotion    	 :  ['Neutral' 'Angry/Serious' 'Happy' 'Sad']
bodypose   	 :  ['Front-Frontish' 'Side' 'Back']
```
##### Data exploration and analysis
First the dataset has been explored to find that there are no null values for any attributes.

There are imbalance in class distribution - specifically emotion and weight have huge imbalance. I'm not going to consider this for the sake of brevity and ease and proceed to network topology and cover it up there.
<!-- (but we can use ...  to handle imbalance in such scenarios) -->

<!-- As I'm not sure on how to handle it now,  not taking this into account and proceed. -->


##### Detailed EDA and preprocessing can be found in this notebook [Click here](#tobedone)

### Implementation details / Approach


#### 1. Selecting Network architecture by trying to overfit the model with 500 images

First aim was to create a tower architecture with enough RF at required blocks.

Any parallel architecture should work since we have to carry forward information from different Receptive fields for different labels.

Initial thought was to go for ResNet(20/34/50/101) for bottom(Network's Base) tower and build label towers(Network's Head) on top of it and tweak the model hyperparameters.
1. Used ResNet50 as base tower and created towers for each label with Dense layers. The time spent on tweaking the network was a considerably high.

1. Conv2D layers without skip connections and Bottleneck layers upto 60 RF, label towers are created with conv layers and GAP

1. Separable Conv layers without skip connections and Bottleneck layers upto 60 RF, label towers are created with Separable Conv layers as well and ended with GAP

1. Simple Bottleneck layers without any skip connections as base tower, after achieving RF of 40+ in base tower, label towers are created with conv layers and GAP

1. Bottleneck layers with skip connections, Conv2D with stride 2 replaced with MaxPooling2D; label towers created with Bottleneck layers with skip connections and output with GAP
> The number of params crossed to over 20 million, hence dropped the idea of having skip connections in label towers  

1. Bottleneck layers with skip connections, Conv2D with stride 2 replaced with MaxPooling2D; label towers created with conv layers and GAP
  - experimented with different blocks and to understand RF of residual - chose the network with less param
> OOM error in base tower with 32 batch size - reduced batch size to 8, steps per epoch will be high and low batch size has negative impact val, train losses

1. Same model as above replaced Conv2D layers with SeparableConv2D layers


> #### All the models were trained on 500 images for 20 epochs with same random_seed and train_test_split. Based on the results and time taken the final model is selected.

##### Created a custom model builder class for finalized model for the data

#### 2. Augmentations

Custom Image data/batch generator has been modified such that
1. It rescales/normalizes all images to scale of 0-1 before stacking for batch
2. It generates augmented images as a separate batch and feed it to the model along with original images.

Keeping this is mind below image augmentations have been experimented with. Separate batches of these augmented images will be fed to the model along with original images. This increases the training data for the model by almost 5 times and reduced the variance.

- Augmentations experimented with
  - horizontal_flip (object positional invariance)
  - blur (quality invariance)
  - brightness_range (brightness invariance)
  - channel_shift_range (shade invariance)
  - cutout(get_random_eraser)  (occlusion invariance)

#### 3. LRfinder and Scheduler/Manager

Used Implementation of One-Cycle Learning rate policy (adapted from Fast.ai lib) from a github repo to find the max-lr

Ran for 5000 iterations with triangular and exp LR Policy. Selected max of max_lr among both of it and used to for custom LR scheduler.

#### Training

Training is done on Google colab and was breaking very frequently due to timeout errors.

#### Visualizations

To do
- Gradcam Visualizations
- Layer Visualizations


## Accuracies after 40 epochs
```
val_gender_output_acc: 0.7185
val_image_quality_output_acc: 0.3376
val_age_output_acc: 0.4154
val_weight_output_acc: 0.6309
val_bag_output_acc: 0.5910
val_footwear_output_acc: 0.6437
val_pose_output_acc: 0.7224
val_emotion_output_acc: 0.6959
```
