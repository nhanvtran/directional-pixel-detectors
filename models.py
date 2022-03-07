from keras.layers import Input, Conv2D, Dense, Dropout, MaxPooling2D, Flatten, concatenate, BatchNormalization
from keras.models import  Model

class MaxPositionResModel:
    @staticmethod
    def build(outputs,
              layerActivation = "relu",
              finalActivation = "linear"):
        # input pre-processing layers
        input1 = Input(shape=(13,21,1), name = "I_input_layer")
        conv1 = Conv2D ( filters = 32, kernel_size = (3 , 3)  , strides = (2 , 2) ,
                         activation = layerActivation , name = "conv_1") ( input1 )
        maxpool1 =  MaxPooling2D((2, 2), name = "maxpool_1")(conv1)
        conv2 = Conv2D ( filters = 64, kernel_size = (3 , 3)  , strides = (2 , 2) ,
                         activation = layerActivation ) ( maxpool1 )
        flattenLayer = Flatten ( ) ( conv2 )
        input2 = Input ( shape = (2 ,) , name = "II_input_layer")
        merge = concatenate ( [flattenLayer , input2] , name = "concat_layer")

        dense1 = Dense ( 32 , activation = layerActivation , name = "dense_1") ( merge )
        dropout1 = Dropout ( 0.1 , name = "dropout_1") ( dense1 )
        output = Dense ( outputs , activation = finalActivation, name = "output_layer" ) ( dropout1 )

        model = Model ( inputs = [input1 , input2] ,
                        outputs = output )
        model.summary ( )

        return model

class CNNBaselineModel:
    @staticmethod
    def build(outputs,
              layerActivation = "relu",
              finalActivation = "linear"):
        # input pre-processing layers
        inputs = Input(shape=(13,21,1), name = "I_input_layer")
        conv1 = Conv2D ( filters = 32, kernel_size = (3 , 3) , strides = (2 , 2) ,
                         activation = layerActivation , name = "conv_1") ( inputs )
        maxpool1 =  MaxPooling2D((2, 2), name = "maxpool_1")(conv1)
        conv2 = Conv2D ( filters = 64 , kernel_size = (3 , 3) , strides = (2 , 2) ,
                         activation = layerActivation , name = "conv_2") ( maxpool1 )
        flattenLayer = Flatten ( ) ( conv2 )
        dense1 = Dense ( 64 , activation = layerActivation , name = "dense_1") ( flattenLayer )
        dropout1 = Dropout ( 0.1 , name = "dropout_1") ( dense1 )
        output = Dense ( outputs , activation = finalActivation, name = "output_layer" ) ( dropout1 )

        model = Model ( inputs = inputs ,
                        outputs = output )
        model.summary ( )

        return model

class CNNDeepModel:
    @staticmethod
    def build(outputs,
              layerActivation = "relu",
              finalActivation = "linear"):
        # input pre-processing layers
        inputs = Input(shape=(13,21,1), name = "input_layer")
        conv1 = Conv2D ( filters = 16, kernel_size = (3 , 3) , strides = (1 , 1) ,
                         activation = layerActivation , name = "conv_1")(inputs)
        conv2 = Conv2D ( filters = 32, kernel_size = (3 , 3) , strides = (1 , 1) ,
                         activation = layerActivation , name = "conv_2") ( conv1 )
        conv3 = Conv2D ( filters = 64 , kernel_size = (3 , 3) , strides = (1 , 1) ,
                         activation = layerActivation, name = "conv_3" ) ( conv2 )
        maxpool1 =  MaxPooling2D((2, 2), name = "maxpool_1")(conv3)
        flattenLayer = Flatten ( ) ( maxpool1 )
        dense1 = Dense ( 64 , activation = layerActivation , name = "dense_1") ( flattenLayer )
        dropout1 = Dropout ( 0.05 , name = "dropout_1") ( dense1 )
        output = Dense ( outputs , activation = finalActivation, name = "output_layer" ) ( dropout1 )

        model = Model ( inputs = inputs ,
                        outputs = output )
        model.summary ( )

        return model

class MultiOutputRegModel:

    def build_default_hidden_layers(self, inputs):
        x = Conv2D ( filters = 16 , kernel_size = (3 , 3) , strides = (1 , 1) ,
                         activation = "relu" , name = "conv_1" ) ( inputs )
        x = Conv2D ( filters = 32 , kernel_size = (3 , 3) , strides = (1 , 1) ,
                         activation = "relu" , name = "conv_2" ) ( x )
        x = MaxPooling2D ( (2 , 2) , name = "maxpool_1" ) ( x )
        x = Conv2D ( filters = 64 , kernel_size = (3 , 3) , strides = (1 , 1) ,
                         activation = "relu" , name = "conv_3" ) ( x )
        x = MaxPooling2D ( (2 , 2) , name = "maxpool_2" ) ( x )

        return x

    def build_coords_branch(self,inputs):

        x = Flatten()(inputs)
        x = Dense(32, activation = "relu")(x)
        # x = BatchNormalization ( ) ( x )
        x = Dense ( 16 , activation =  "relu") ( x )
        x = Dropout ( 0.1 ) ( x )
        x = Dense ( 2, activation = "linear", name = "coords_output" ) ( x )
        return x

    def build_angles_branch(self,inputs):
        x = Flatten()(inputs)
        x = Dense ( 32 , activation =  "relu") ( x )
        # x = BatchNormalization ( ) ( x )
        x = Dropout ( 0.1 ) ( x )
        x = Dense ( 2, activation = "linear", name = "angles_output" ) ( x )
        return x

    def assemble_full_model(self,inputs_shape):
        inputs = Input ( shape = inputs_shape )
        common_branch_inputs = self.build_default_hidden_layers ( inputs )
        coords_output = self.build_coords_branch(common_branch_inputs)
        angles_output = self.build_angles_branch(common_branch_inputs)
        output = concatenate ( [coords_output, angles_output] , name = "total_output")
        model = Model ( inputs = inputs ,
                        outputs = output,
                        name = "model_1" )
        model.summary()
        return model