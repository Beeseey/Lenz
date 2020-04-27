'''
get the dataset,
get features
get the model to extract dataset
if features do not exists run predict.py
get save features or not
get csv concept file
train model
save model
get test data
generate f1 or bleu
'''
import sys
import numpy as np
from tqdm import tqdm
sys.path.append("..")

from tensorflow.keras.preprocessing.sequence import pad_sequences
from extract import binary_encoding

def generator(model,tokenizer, max_length, vocab_size, data, 
	batch_Size, use_generator):

	X1, X2, y = list(), list(), list()
	count = 0
	while True:
		for data_object in data:
			X1_, X2_, y_ = create_sequences(model,tokenizer, 
				max_length, vocab_size, [data_object],use_generator)
			X1.extend(X1_)
			X2.extend(X2_)
			y.extend(y_)
			count+=1
			
			if count == batch_Size:
				yield [np.array(X1), np.array(X2)], np.array(y)
				X1, X2, y = list(), list(), list()
				count = 0
			'''
			X1.append(in_img[0])
			X2.append(in_seq[0])
			y.append(out_seq[0])
			for i in range

			if len(X1) == batch_Size:
				yield np.array(X1), np.array(X2), np.array(y)
				X1, X2, y = list(), list(), list()
			'''

def create_sequences(model,tokenizer, max_length, vocab_size, 
	data, use_generator):
	X1, X2, y = list(), list(), list()
	# walk through each data object
	if use_generator is False:
		data = tqdm(data)
		data.set_description("Creating Sequences")
	for data_object in data:
		
		image_feature = data_object['image_feature']
		concepts = data_object['concepts']
		concepts = ' '.join(concepts)
		concepts = 'startseq '+concepts+' endseq'
		seq = tokenizer.texts_to_sequences([concepts])[0]
		
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			if model is not None:
				in_seq = in_seq.reshape(1,in_seq.shape[0],1)
				in_seq = model.predict(in_seq)
				in_seq = np.squeeze(in_seq,axis=0)
			out_seq = binary_encoding(vocab_size,[out_seq])
			#out_seq = np.expand_dims(out_seq)

			X1.append(image_feature)
			X2.append(in_seq)
			y.append(out_seq)
	#print(array(X1).shape,'x1')
	#print(array(X2).shape,'x2')
	#print(array(y).shape,'y')
	
	return np.array(X1), np.array(X2), np.array(y)


def train(LSTM=None,text_encoder=None,train_data=None,test_data=None,
	EPOCHS=None,OPT=None,use_generator=False,BATCH_SIZE=None,
	VERBOSE=None,Checkpoint=None,tokenizer=None,max_length=None,
	vocab_size=None):

	if use_generator:
		train_gen = generator(text_encoder, 
			tokenizer, max_length, vocab_size,train_data,BATCH_SIZE, use_generator)
		test_gen = generator(text_encoder, 
			tokenizer, max_length, vocab_size,train_data,BATCH_SIZE,use_generator)

		LSTM.fit(train_gen,epochs=EPOCHS, verbose=VERBOSE, 
			callbacks=[Checkpoint],validation_data=test_gen, 
			steps_per_epoch=len(train_data)//BATCH_SIZE,
			validation_steps=len(test_data)//BATCH_SIZE)

	else:
		X1train, X2train, ytrain = create_sequences(text_encoder, 
			tokenizer, max_length, vocab_size,train_data,use_generator)
		X1test, X2test, ytest = create_sequences(text_encoder,
			tokenizer, max_length, vocab_size, test_data,use_generator)

		print(X1train.shape,X2train.shape,ytrain.shape)

		print(LSTM.summary())

		LSTM.fit([X1train, X2train], ytrain, epochs=EPOCHS,
			verbose=VERBOSE, callbacks=[Checkpoint],
			validation_data = ([X1test, X2test], ytest))