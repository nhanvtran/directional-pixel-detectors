from keras.models import Model
from keras.layers import (BatchNormalization, Flatten, Input, Reshape, Dense, MaxPool2D,
                          Conv1D, MaxPool1D, Conv2D, Dropout, concatenate)

class RegModelCotAlpha1D:
    
    def build(self,inputs):
        x = Conv1D (filters = 64, kernel_size = 3, strides=2, activation="relu",
                   kernel_initializer = "he_normal")(inputs)
        x = BatchNormalization()(x)
        x = MaxPool1D(pool_size= 2)(x)
        x = Conv1D (filters = 32, kernel_size = 3, strides=2, activation="relu",
                   kernel_initializer = "he_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = 0.2)(x)
        x = Flatten()(x)
        x = Dense(16, kernel_initializer = "he_normal", activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = 0.2)(x)
        x = Dense(8, kernel_initializer = "he_normal", activation='relu')(x)
        x = BatchNormalization()(x)       
        alpha_output = Dense(1, activation = "linear")(x)
        return alpha_output

        return x
 
    def assemble_model(self):
        profile_inputs = Input ( shape = (21,1) )
        alpha_output = self.build(profile_inputs)
        model = Model(inputs = profile_inputs, outputs=alpha_output, name = "cotAlpha_1D_model")
        print( model.summary() )
        return model
    

class RegModelCotAlpha2D:

    def build_image_branch(self,inputs):
        x = Conv2D(128, (2, 2), kernel_initializer = "he_normal",
                   strides=(2, 2), activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (2, 2), kernel_initializer = "he_normal",
           strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = 0.2)(x)
        x = Flatten()(x)
        x = Dense(32, kernel_initializer = "he_normal", activation='relu', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = Dense(16, kernel_initializer = "he_normal", activation='relu', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = Dense(1, activation= "linear", name = "final_output")(x)

        return x
 
    def assemble_model(self):
        inputs = Input ( shape = (12,21,1) )
        outputs = self.build_image_branch(inputs)
        model = Model(inputs =inputs, outputs=outputs, name = "cotAlpha_2D_x_profile_model")
        print( model.summary() )
        return model
    

class RegModelCotAlphaIM2D:

    def build_image_branch(self,inputs):
        x = Conv2D(256, (2, 2), kernel_initializer = "he_normal",
                   strides=(2, 2), activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(rate = 0.)(x)
        x = Conv2D(128, (2, 2), kernel_initializer = "he_normal",
           strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = 0.2)(x)
        x = Conv2D(64, (2, 2), kernel_initializer = "he_normal",
           strides=(2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = 0.2)(x)
        x = Flatten()(x)
        x = Dense(32, kernel_initializer = "he_normal", activation='relu', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = Dense(16, kernel_initializer = "he_normal", activation='relu', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = Dense(1, activation= "linear", name = "final_output")(x)

        return x
 
    def assemble_model(self):
        inputs = Input ( shape = (13,21,1) )
        outputs = self.build_image_branch(inputs)
        model = Model(inputs =inputs, outputs=outputs, name = "cotAlpha_2D_im_model")
        print( model.summary() )
        return model
    

class RegModelCotAlpha3D:

    def build_image_branch(self,inputs):
        x = Conv2D(256, (3, 3), kernel_initializer = "he_normal",
                   strides=(2, 2), activation='relu',
                   data_format = "channels_first")(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(128, (2, 2), kernel_initializer = "he_normal",
                   strides=(2, 2), activation='relu',
                   data_format = "channels_first")(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), kernel_initializer = "he_normal",
                   strides=(2, 2), activation='relu',
                   data_format = "channels_first")(x)
        x = BatchNormalization()(x)
#         x = Conv2D(32, (2, 2), kernel_initializer = "he_normal",
#                    strides=(2, 2), activation='relu',
#                    data_format = "channels_first")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate = 0.25)(x)
        x = Flatten()(x)
        x = Dense(32, kernel_initializer = "he_normal", activation='relu', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = Dense(16, kernel_initializer = "he_normal", activation='relu', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = Dense(1, activation= "linear", name = "final_output")(x)

        return x
 
    def assemble_model(self):
        inputs = Input ( shape = (5,13,21) )
        outputs = self.build_image_branch(inputs)
        model = Model(inputs =inputs, outputs=outputs, name = "cotAlpha_3D_model")
        print( model.summary() )
        return model