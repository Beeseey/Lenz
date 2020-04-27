from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, TimeDistributed, Bidirectional, concatenate, Lambda, dot, Activation
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import Adam

def attention_3d_block(hidden_states):

    # @author: felixhao28.

    # hidden_states.shape = (batch_size, time_steps, hidden_size)

    hidden_size = int(hidden_states.shape[2])

    # Inside dense layer

    #              hidden_states            dot               W            =>           score_first_part

    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)

    # W is the trainable weight matrix of attention Luong's multiplicative style score

    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)

    #            score_first_part           dot        last_hidden_state     => attention_weights

    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)

    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)

    score = dot([score_first_part, h_t], [2, 1], name='attention_score')

    attention_weights = Activation('softmax', name='attention_weight')(score)

    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)

    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')

    pre_activation = concatenate([context_vector, h_t], name='attention_output')

    attention_vector = Dense(256, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)

    return attention_vector

def AlternativeRNNModel(image_feat_shape,vocab_size, max_len):
	
	embedding_size = 256
	
	input_1 = Input(shape=(image_feat_shape,))
	#inp1 = baseModel.output

	#print(inp1.shape)
	#headModel = Flatten()(inp1)
	#print(headModel.shape)
	#headModel = Dense(512*7*7, activation="relu")(headModel)
	headModel = Dense(512, activation="relu")(input_1)
	headModel = Dropout(0.5)(headModel)
	#headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)
	
	#image_input = Input(shape=(1000,))
	image_model_1 = Dense(embedding_size, activation='relu')(headModel)
	image_model = RepeatVector(max_len)(image_model_1)
	#print(image_model.shape)

	caption_input = Input(shape=(max_len,))
	# mask_zero: We zero pad inputs to the same length, the zero mask ignores those inputs. E.g. it is an efficiency.
	caption_model_1 = Embedding(vocab_size, embedding_size, mask_zero=True)(caption_input)
	# Since we are going to predict the next word using the previous words
	# (length of previous words changes with every iteration over the caption), we have to set return_sequences = True.
	caption_model_2 = LSTM(256, return_sequences=True)(caption_model_1)
	# caption_model = TimeDistributed(Dense(embedding_size, activation='relu'))(caption_model_2)
	caption_model = TimeDistributed(Dense(embedding_size))(caption_model_2)

	# Merging the models and creating a softmax classifier
	final_model_1 = concatenate([image_model, caption_model])
	# final_model_2 = LSTM(rnnConfig['LSTM_units'], return_sequences=False)(final_model_1)

	final_model_2 = Bidirectional(LSTM(256, return_sequences=True))(final_model_1)
	attention_output = attention_3d_block(final_model_2)
	# final_model_3 = Dense(rnnConfig['dense_units'], activation='relu')(final_model_2)
	# final_model = Dense(vocab_size, activation='softmax')(final_model_3)
	final_model = Dense(vocab_size, activation='softmax')(attention_output)

	model = Model(inputs=[input_1, caption_input], outputs=final_model)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model

def AlternativeRNNModel2(vocab_size, max_len):
	
	embedding_size = 256
	
	input_1 = Input(shape=(256,))

	#image_model_1 = Dense(embedding_size, activation='relu')(input_1)
	image_model = RepeatVector(max_len)(input_1)

	caption_input = Input(shape=(max_len,))
	# mask_zero: We zero pad inputs to the same length, the zero mask ignores those inputs. E.g. it is an efficiency.
	caption_model_1 = Embedding(vocab_size, embedding_size, mask_zero=True)(caption_input)
	# Since we are going to predict the next word using the previous words
	# (length of previous words changes with every iteration over the caption), we have to set return_sequences = True.
	caption_model_2 = LSTM(256, return_sequences=True)(caption_model_1)
	# caption_model = TimeDistributed(Dense(embedding_size, activation='relu'))(caption_model_2)
	caption_model = TimeDistributed(Dense(embedding_size))(caption_model_2)

	# Merging the models and creating a softmax classifier
	final_model_1 = concatenate([image_model, caption_model])
	# final_model_2 = LSTM(rnnConfig['LSTM_units'], return_sequences=False)(final_model_1)

	final_model_2 = LSTM(256, return_sequences=True)(final_model_1)
	attention_output = attention_3d_block(final_model_2)
	# final_model_3 = Dense(rnnConfig['dense_units'], activation='relu')(final_model_2)
	# final_model = Dense(vocab_size, activation='softmax')(final_model_3)
	final_model = Dense(vocab_size, activation='softmax')(attention_output)

	model = Model(inputs=[input_1, caption_input], outputs=final_model)

	#print(model.summary())
	# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model

def AlternativeRNNModel3(vocab_size):
	
	input_1 = Input(shape=(256,))

	input_2 = Input(shape=(256,))
	
	merge_input = concatenate([input_1, input_2])
	# final_model_2 = LSTM(rnnConfig['LSTM_units'], return_sequences=False)(final_model_1)

	#final_model = LSTM(256, return_sequences=True)(merge_input)
	#attention_output = attention_3d_block(final_model)
	# final_model_3 = Dense(rnnConfig['dense_units'], activation='relu')(final_model_2)
	# final_model = Dense(vocab_size, activation='softmax')(final_model_3)
	final_model = Dense(vocab_size, activation='sigmoid')(merge_input)

	model = Model(inputs=[input_1, input_2], outputs=final_model)

	#print(model.summary())
	# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model