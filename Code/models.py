import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Conv1D, BatchNormalization, MaxPool1D, LayerNormalization, Dropout, Dense, Flatten
from tfomics.layers import MultiHeadAttention

class Activation(Layer):
    
    def __init__(self, func, name='conv_activation'):
        super(Activation, self).__init__(name=name)
        
        funcs = {
            'relu' : self.relu,
            'exp' : self.exp,
            'softplus' : self.softplus,
            'gelu' : self.gelu,
            'sigmoid' : self.modified_sigmoid
        }
        self.func = funcs[func]
        
    def call(self, inputs):
        return self.func(inputs)
        
    def relu(self, inputs):
        return tf.math.maximum(0., inputs)
    
    def exp(self, inputs):
        return tf.math.exp(inputs)
        
    def softplus(self, inputs):
        return tf.math.log(1 + tf.math.exp(inputs))
    
    def gelu(self, inputs):
        return 0.5 * inputs * (1.0 + tf.math.erf(inputs / tf.cast(1.4142135623730951, inputs.dtype)))
    
    def modified_sigmoid(self, inputs):
        return 10 * tf.nn.sigmoid(inputs - 8)
    
    
def CNN_ATT(in_shape=(200, 4), filters=32, kernel_size=19, batch_norm=True, activation='relu', pool_size=4, layer_norm=True, heads=8, vector_size=32, layer_norm2=False, dense_units=512, num_out=12):
    
    inputs = Input(shape=in_shape)
    
    nn = Conv1D(filters=filters, kernel_size=kernel_size, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = BatchNormalization()(nn)
    nn = Activation(activation)(nn)
    nn = MaxPool1D(pool_size)(nn)
    nn = Dropout(0.1)(nn)
    
    if layer_norm:
        nn = LayerNormalization()(nn)
    nn, att = MultiHeadAttention(num_heads=heads, d_model=heads*vector_size)(nn, nn, nn)
    if layer_norm2:
        nn = LayerNormalization()(nn)
    nn = Dropout(0.1)(nn)
    
    nn = Flatten()(nn)
    nn = Dense(dense_units, use_bias=False)(nn)
    nn = BatchNormalization()(nn)
    nn = Activation('relu', name='dense_activation')(nn)
    nn = Dropout(0.5)(nn)
    outputs = Dense(num_out, activation='sigmoid')(nn)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model



















