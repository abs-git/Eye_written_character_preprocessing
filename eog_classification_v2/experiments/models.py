import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Lambda, LayerNormalization
from tensorflow.keras.activations import softmax

from attention import Attention


class ViTBaseModel():
    def __init__(self, input_shape, params):
        self.input_shape = input_shape
        self.batch_size = params['batch_size']

        self.hidden_size = params['hidden_size']
        self.heads = params['heads']
        self.dropout = params['dropout']
        self.mlp_dropout = params['mlp_dropout']
        self.mlp_units = params['mlp_units']
        self.n_layers = params['n_layers']
        self.patch_size = params['patch_size']

    def mlp(self, input):
        output_dim = input.shape[-1]
        x = input
        for dim in self.mlp_units:
            x = Dense(dim)(x)
            x = Dropout(self.mlp_dropout)(x)
        out = Dense(output_dim)(x)
        return out


    def msa(self, input):
        batch_size = input.shape[0]
        projection_dim = self.hidden_size // self.heads

        q = Dense(self.hidden_size)(input)
        q = tf.reshape(q, (batch_size, -1, self.heads, projection_dim))
        k = Dense(self.hidden_size)(input)
        k = tf.reshape(k, (batch_size, -1, self.heads, projection_dim))
        v = Dense(self.hidden_size)(input)
        v = tf.reshape(v, (batch_size, -1, self.heads, projection_dim))

        score = tf.matmul(q,k, transpose_b=True)
        score = score / k.shape[-1]**(1/2)
        weights = softmax(score)
        attn_score = tf.matmul(weights, v)

        out = tf.transpose(attn_score, perm=[0,2,1,3])
        out = tf.reshape(out, (batch_size, -1, self.hidden_size))
        out = Dense(self.hidden_size)(out)

        return out


    def encoder_block(self, input):
        res = input
        x = LayerNormalization(epsilon=1e-6)(input)
        x = self.msa(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(self.dropout)(x)
        x += res

        out = self.mlp(x)
        out = LayerNormalization(epsilon=1e-6)(x)
        return out


    def transformer_encoder(self, input):
        x = input
        for i in range(self.n_layers):
            x = self.encoder_block(x)
        
        out = LayerNormalization(epsilon=1e-6)(x)
        return out


    def model(self):
        input_ = Input(shape = self.input_shape)        # (None, length, points)

        emb = Conv1D(filters=self.hidden_size, kernel_size=(self.patch_size,), strides=(self.patch_size,))(input_)  # patches : (None, length, points) -> (None, n_patch, dim)
        emb = tf.reshape(emb, [self.batch_size, emb.shape[1], emb.shape[2]])                                        # (None, n_patch, dim) -> (batch_size, n_patch, dim))

        cls_tokens = tf.Variable(initial_value=tf.random.normal([1, self.hidden_size]))             # (1, dim)
        cls_tokens = tf.convert_to_tensor([cls_tokens]*emb.shape[0])                                # (batch_size, 1, dim)
        x = tf.concat([cls_tokens, emb], axis=1)        # (batch_size, cls_tokens + n_patch, dim)

        pos_embedding = tf.Variable(initial_value=tf.random.normal([1, x.shape[1], self.hidden_size]))      # (1, cls_tokens + n_patch, dim)
        x += pos_embedding                              # (batch_size, cls_tokens + n_patch, dim)

        x = self.transformer_encoder(x)                 # (batch_size, cls_tokens + n_patch, output_dim)
        
        output_ = x[:,0,:]                              # (batch_size, cls_token, output_dim)

        model_ = Model(inputs = input_, outputs = output_)
        return model_


class HybridBaseModel():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def model(self):
        input_ = Input(self.input_shape, name = "Input")

        X = Conv1D(64, kernel_size = 3, strides = 1, activation='relu',name = "Conv_1")(input_)
        X = BatchNormalization(name = "Conv_1_Batch_norm")(X)
        X = Dropout(rate = 0.2, name = "Conv_1_Dropout")(X)

        X = Conv1D(64, kernel_size = 3, strides = 1, activation='relu',name = "Conv_2")(X)      
        X = BatchNormalization(name = "Conv_2_Batch_norm")(X)
        X = Dropout(rate = 0.2,name = "Conv_2_Dropout")(X)

        X = LSTM(units = 64, return_sequences = True, name = "LSTM_1")(X)
        X = BatchNormalization(name = "LSTM_1_Batch_norm")(X)
        X = Dropout(rate = 0.2,name = "LSTM_1_Dropout")(X)
        
        X = LSTM(units = 64, return_sequences = True, name = "LSTM_2")(X)
        X = Attention(32)(X)
        X = BatchNormalization(name = "LSTM_2_Batch_norm")(X)
        X = Dropout(rate = 0.2,name = "LSTM_2_Dropout")(X)
        
        output_ = Dense(128)(X)

        model_ = Model(inputs = input_, outputs = output_)

        return model_


def binary_siamese_net(input_shape, base_model):
  left_input = Input(input_shape, name= "Left Input")
  right_input = Input(input_shape, name= "Right Input")

  encoded_left = base_model(left_input)
  encoded_right = base_model(right_input)

  L1_layer = Lambda(lambda tensor: K.abs(tensor[0]-tensor[1]))
  L1_distance = L1_layer([encoded_left,encoded_right])

  prediction = Dense(1, activation = 'sigmoid',name = "Dense")(L1_distance)

  model = Model(inputs = [left_input, right_input], outputs = prediction)

  return model