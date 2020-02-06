from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D  
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


class MultiLabelCustomResnetBuilder(object):
    # Adapted from below source
    # https://github.com/raghakot/keras-resnet/blob/master/resnet.py
    def __init__(self, input_shape):
        # self.input_tensor = input_tensor
        # self.input_shape = K.int_shape(input_tensor)
        self.input_shape = input_shape
        self.output = self.build()

    def get_model(self):
        return self.output

    def build(self):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        
        input_tensor = Input(shape=self.input_shape)

        conv1 = self._conv_bn_relu(filters=32, kernel_size=(3, 3), padding="same")(input_tensor)
        conv2 = self._conv_bn_relu(filters=64, kernel_size=(3, 3), padding="same", dropout=0.05)(conv1)

        filters = 32

        backbone = self.build_resnet_tower(conv2, filters=filters, block_repititions=[3, 3])

        gender_activation = "softmax"
        image_quality_activation = "softmax"
        age_activation ="softmax"
        weight_activation = "softmax"
        bag_activation = "softmax"
        footwear_activation = "softmax"
        emotion_activation = "softmax"
        pose_activation = "softmax"

        filters = 32

        # heads
        gender = self.build_label_tower(backbone, "gender", filters=filters, num_outputs=2, activation=gender_activation)

        image_quality = self.build_label_tower(backbone, "image_quality", filters=filters, num_outputs=3, activation=image_quality_activation)

        age = self.build_label_tower(backbone, "age", filters=filters,  num_outputs= 5, activation=age_activation)

        weight = self.build_label_tower(backbone, "weight", filters=filters,  num_outputs= 4, activation=weight_activation)

        bag = self.build_label_tower(backbone, "bag", filters=filters,  num_outputs= 3, activation=bag_activation)

        footwear = self.build_label_tower(backbone, "footwear", filters=filters,  num_outputs= 3, activation=footwear_activation)

        pose = self.build_label_tower(backbone, "pose", filters=filters,  num_outputs= 3, activation=pose_activation)

        emotion = self.build_label_tower(backbone, "emotion", filters=filters,  num_outputs= 4, activation=emotion_activation)


        model = Model(
            inputs=input_tensor, 
            outputs=[gender, image_quality, age, weight, bag, footwear, pose, emotion]
        )
        return model


    def build_label_tower(self, input_tensor, name, filters, num_outputs, activation): # input 56
        tower_name = f"{name}_tower"
        block_input = Dropout(0.15, name=tower_name)(input_tensor)
#         for tower_num, tower_size in enumerate(block_repititions):
#             for block in range(tower_size):
#                     init_strides = (1, 1)
#                     if tower_num == 0 and block == 0:
#                         is_first_layer_of_first_block = True
#                         filters *= 2
#                     elif block == 0:
# #                     Not using stride of 2 to avoid checker board, but using maxpool for all blocks
# #                         init_strides = (2, 2)
#                         is_first_layer_of_first_block = False
#                         block_input = MaxPooling2D(2)(block_input)
#                         filters *= 2
#                     block_input = self.bottleneck_block(filters, init_strides)(block_input)
#         # Last activation
#         tower = self._bn_relu(block_input)
        
        conv_1 = Conv2D(filters, 1, activation='relu', kernel_initializer="he_normal", name=name+'_c1')(block_input)
        conv_2 = self._bn_relu_conv(filters=filters*2, kernel_size=(3, 3), padding='valid')(conv_1)
        conv_2 = Dropout(0.5)(conv_2)
        conv_2 = MaxPooling2D(2)(conv_2)
        conv_3 = self._bn_relu_conv(filters=filters*4, kernel_size=(3, 3), padding='valid')(conv_2)
        conv_3 = Dropout(0.1)(conv_3)
        conv_3 = MaxPooling2D(2)(conv_3)
        conv_4 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3), padding='valid')(conv_3)
        conv_4 = Dropout(0.1)(conv_4)
        final_conv = Conv2D(num_outputs, 1)(conv_4)
        gap = GlobalAveragePooling2D()(final_conv)
        output = Activation(activation=activation, name=f"{name}_output")(gap)
        return output


    def build_resnet_tower(self, input_tensor, filters, block_repititions):
        block_input = input_tensor
        for tower_num, tower_size in enumerate(block_repititions):
            for block in range(tower_size):
                    init_strides = (1, 1)
                    if tower_num == 0 and block == 0:
                        is_first_layer_of_first_block = True
                        filters *= 2
                    elif block == 0:
#                     Not using stride of 2 to avoid checker board, but using maxpool for all blocks
#                         init_strides = (2, 2)
                        is_first_layer_of_first_block = False
                        block_input = MaxPooling2D(2)(block_input)
                        filters *= 2
                    block_input = self.bottleneck_block(filters, init_strides)(block_input)
        # Last activation
        tower = self._bn_relu(block_input)
        return tower


    def _bn_relu(self, input):
        """Helper to build a BN -> relu block
        """
        norm = BatchNormalization()(input)
        return Activation("relu")(norm)


    def _conv_bn_relu(self, **kwargs):
        """Helper to build a conv -> BN -> relu block
        """
        filters = kwargs["filters"]
        kernel_size = kwargs["kernel_size"]
        strides = kwargs.setdefault("strides", (1, 1))
        kernel_initializer = kwargs.setdefault("kernel_initializer", "he_normal")
        padding = kwargs.setdefault("padding", "same")
        kernel_regularizer = kwargs.setdefault("kernel_regularizer", l2(1.e-4))
        dropout = kwargs.setdefault("dropout", False)

        def f(input): # to be able to take input layer as input and return back layer
            conv = Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer)(input)
        #                   ,kernel_regularizer=kernel_regularizer)(input)
            if dropout:
                actv = self._bn_relu(conv)
                return Dropout(dropout)(actv)

            return self._bn_relu(conv)
        return f



    def _bn_relu_conv(self, **kwargs):
        """Helper to build a BN -> relu -> conv block.
        This is an improved scheme proposed in ResNetV2 http://arxiv.org/pdf/1603.05027v2.pdf
        """
        filters = kwargs["filters"]
        kernel_size = kwargs["kernel_size"]
        strides = kwargs.setdefault("strides", (1, 1))
        kernel_initializer = kwargs.setdefault("kernel_initializer", "he_normal")
        padding = kwargs.setdefault("padding", "same")
        kernel_regularizer = kwargs.setdefault("kernel_regularizer", l2(1.e-4))
        dropout = kwargs.setdefault("dropout", False)

        def f(input):

            activation = self._bn_relu(input)
            if dropout:
                conv = Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer)(activation)
                return Dropout(dropout)(conv)

            return Conv2D(filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          kernel_initializer=kernel_initializer)(activation)
        #                   ,kernel_regularizer=kernel_regularizer)(activation)
        return f


    def bottleneck_block(self, filters, name='',init_strides=(1, 1), is_first_layer_of_first_block=False):
        """Bottleneck architecture for deeper resnet.
        Follows improved proposed scheme in ResNetV2 http://arxiv.org/pdf/1603.05027v2.pdf
        """
        def f(input): 
            if is_first_layer_of_first_block:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                  strides=init_strides,
                                  padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=l2(1e-4), name=name)(input)
            else:
                conv_1_1 = self._bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                         strides=init_strides, name=name)(input)

            conv_3_3 = self._bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
            residual = self._bn_relu_conv(filters=filters, kernel_size=(1, 1))(conv_3_3)

            return self._add_shortcut(input, residual)

        return f
    
    def _add_shortcut(self, input, residual):
        """Adds a shortcut between input and residual block and merges them with "sum"
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.

        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)

        if K.common.image_dim_ordering() == 'tf':
            row_axis = 1
            col_axis = 2
            channel_axis = 3
        else:
            channel_axis = 1
            row_axis = 2
            col_axis = 3

        stride_width = int(round(input_shape[row_axis] / residual_shape[row_axis]))
        stride_height = int(round(input_shape[col_axis] / residual_shape[col_axis]))
        equal_channels = input_shape[channel_axis] == residual_shape[channel_axis]
        # print('input_shape[row_axis], residual_shape[row_axis] :', input_shape[row_axis] , residual_shape[row_axis])
        # print('input_shape[col_axis], residual_shape[col_axis] :', input_shape[col_axis] , residual_shape[col_axis])
        # print('input_shape[channel_axis], residual_shape[channel_axis] :', input_shape[channel_axis] , residual_shape[channel_axis])
        # print('row_axis, col_axis, channel_axis : ', row_axis, col_axis, channel_axis)
        shortcut = input

        if stride_width > 1 or stride_height > 1 or not equal_channels:
        # 1 X 1 conv if shape is different. Else identity.
            shortcut = Conv2D(filters=residual_shape[channel_axis],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)

        return add([shortcut, residual])


def get_custom_model(input_shape=(224, 224, 3)):
    model = MultiLabelCustomResnetBuilder(input_shape)
    return model.get_model()
    