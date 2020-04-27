from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv1D, MaxPool1D, UpSampling1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.layers import Embedding, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

def auto_encoder_text2(vocab_size=None,filters=None,latentDim=None):

	input_layer = Input(shape=(vocab_size, 1))
	x = input_layer

	for f in filters:
		x = Conv1D(f, (3), activation='relu', padding='same')(x)
		x = MaxPool1D( (2), padding='same')(x)

	volumeSize = K.int_shape(x)

	x = Flatten(name="bottleneck")(x)

	latent = Dense(latentDim, activation='tanh', name='encoded')(x)
	x = Dense(np.prod(volumeSize[1:]))(latent)

	x = Reshape((volumeSize[1], volumeSize[2]))(x)

	# decoding architecture
	for f in filters[::-1]:
		x = Conv1D(f, (3), activation='relu', padding='causal')(x)
		x = UpSampling1D((2))(x)
	
	output_layer   = Conv1D(1, (3), padding='same', activation='sigmoid')(x)

	# compile the model
	model = Model(input_layer, output_layer)
	model.compile(optimizer='adam', loss='mse')
	
	return model

def auto_encoder_text(maxlen=None,vocab_size=None,embedding_size=None,
	latent_dim=None):

	caption_input = Input(shape=(maxlen,), name='Encoder-Input')
	caption_model_1 = Embedding(vocab_size, embedding_size, 
		mask_zero=True)(caption_input)

	state_h = LSTM(latent_dim,activation='relu'
		,name='encoded')(caption_model_1)
	#decoded = RepeatVector(maxlen)(state_h)
	#decoder_lstm = LSTM(latent_dim, activation='relu',
	#	return_sequences= True,name='Decoder-LSTM-before')(decoded)
	#decoder_lstm = Flatten()(decoder_lstm)
	decoder_dense = Dense(vocab_size*10, activation='relu')(state_h)
	decoder_dense = Dense(vocab_size, activation='sigmoid', 
		name='Final-Output-Dense-before')(decoder_dense)

	# compile the model
	model = Model(caption_input, decoder_dense)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	
	return model

def auto_encoder(width=None, height=None, depth=None, filters=None, 
	latentDim=None):
	# initialize the input shape to be "channels last" along with
	# the channels dimension itself
	# channels dimension itself
	inputShape = (height, width, depth)
	chanDim = -1
	# define the input to the encoder
	inputs = Input(shape=inputShape)
	x = inputs
	# loop over the number of filters
	for f in filters:
		# apply a CONV => RELU => BN operation
		x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = BatchNormalization(axis=chanDim)(x)
	# flatten the network and then construct our latent vector
	volumeSize = K.int_shape(x)
	x = Flatten()(x)
	latent = Dense(latentDim, name="encoded")(x)
	x = Dense(np.prod(volumeSize[1:]))(latent)
	x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
	# loop over our number of filters again, but this time in
	# reverse order
	for f in filters[::-1]:
		# apply a CONV_TRANSPOSE => RELU => BN operation
		x = Conv2DTranspose(f, (3, 3), strides=2,
			padding="same")(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = BatchNormalization(axis=chanDim)(x)
	# apply a single CONV_TRANSPOSE layer used to recover the
	# original depth of the image
	x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
	outputs = Activation("sigmoid", name="decoded")(x)
	# construct our autoencoder model
	autoencoder = Model(inputs, outputs, name="autoencoder")
	# return the autoencoder model
	autoencoder.compile(optimizer='adam', loss='mse')

	return autoencoder