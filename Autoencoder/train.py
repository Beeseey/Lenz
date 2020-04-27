import sys
import numpy as np
import cv2
import pickle
from tqdm import tqdm
sys.path.append("..")

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from extract import binary_encoding

from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True 

def generator(text_or_image,tokenizer, max_length, vocab_size, data, 
	batch_Size, use_generator, input_size):

	X1, y = list(), list()
	count = 0
	while True:
		for data_object in data:
			X1_, y_ = create_sequences(text_or_image, tokenizer, 
				max_length, vocab_size, [data_object],use_generator, input_size)
			X1.extend(X1_)
			y.extend(y_)
			count+=1
			
			if count == batch_Size:
				yield np.array(X1), np.array(y)
				X1, y = list(), list()
				count = 0

def create_sequences(text_or_image,tokenizer, max_length, vocab_size, 
	data,use_generator, input_size):
	x, y = list(), list()
	# walk through each data object
	if use_generator is False:
		data = tqdm(data)
		data.set_description("Creating Sequences")

	for data_object in data:
		
		if text_or_image == 'text':
			concepts = data_object['concepts']
			concepts = ' '.join(concepts)
			#concepts = 'startseq '+concepts+' endseq'
			seq = tokenizer.texts_to_sequences([concepts])[0]
		
			in_seq = seq[:]
			out_seq = in_seq
		
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		
			out_seq = binary_encoding(vocab_size,[out_seq])
			#out_seq = to_categorical([in_seq], num_classes=vocab_size)[0]
			#out_seq = np.expand_dims(out_seq)
			#print(out_seq.shape)
			x.append(in_seq)
			y.append(out_seq)
		elif text_or_image == 'image':
			path = data_object['filepath']
			image = load_image(path,input_size)
			x.append(image)
			y.append(image)
	
	return np.array(x), np.array(y)

def load_image(path, target_size):
	
	try:
		image = load_img(path, target_size=target_size)
		# convert the image pixels to a numpy array
		image = img_to_array(image)
	except:
		image = cv2.imread(path)
		image =  cv2.resize(image, target_size, interpolation = cv2.INTER_AREA)
		image = Image.fromarray(image)
		image = img_to_array(image)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#image = np.expand_dims(image,axis=-1)
	image = image.astype("float32")/255.0
	return image
'''
data_list,concept_list,tokenizer,vocab_size = load_data('../RNN_concept_model/dense_121_feat.pkl')

dataset_number = len(data_list)
train_data = data_list[:int(0.85*dataset_number)]
test_data = data_list[int(0.85*dataset_number):]
'''
def train(autoencoder=None,text_or_image=None,train_data=None,test_data=None,
	EPOCHS=None,OPT=None,use_generator=False,BATCH_SIZE=None,
	VERBOSE=None,Checkpoint=None,tokenizer=None,max_length=None,
	vocab_size=None,image_input_size=None):
	
	if use_generator:
		
		train_gen = generator(text_or_image, tokenizer, 142, vocab_size, 
			train_data,BATCH_SIZE,True,image_input_size)
		test_gen = generator(text_or_image, tokenizer, 142, vocab_size, 
			test_data,BATCH_SIZE,True,image_input_size)

		autoencoder.fit(train_gen,epochs=EPOCHS, verbose=VERBOSE, 
			callbacks=[Checkpoint],validation_data=test_gen, 
			steps_per_epoch=len(train_data)//BATCH_SIZE,
			validation_steps=len(test_data)//BATCH_SIZE)

	else:
		train_x, train_y = create_sequences(text_or_image, tokenizer, max_length, 
			vocab_size,train_data,use_generator,image_input_size)
		test_x, test_y = create_sequences(text_or_image,tokenizer, max_length, 
			vocab_size, test_data,use_generator,image_input_size)

		print(autoencoder.summary())

		autoencoder.fit(train_x, train_y, epochs=EPOCHS,
			verbose=VERBOSE, callbacks=[Checkpoint],
			validation_data = (test_x, test_y))

	return autoencoder

'''
train_X,train_Y = create_sequences(tokenizer, 142, vocab_size, 
	train_data,False)
test_X,test_Y = create_sequences(tokenizer, 142, vocab_size, 
	test_data,False)

print(train_X.shape)
print(train_Y.shape)
print(test_X.shape)
print(test_Y.shape)

model = auto_encoder_text2(maxlen=142,vocab_size=vocab_size,
	embedding_size=256,latent_dim=256)

model.fit(train_gen, epochs=1,
			verbose=1, steps_per_epoch=len(train_data)//100 ,
			validation_data = test_gen, 
			validation_steps=len(test_data)//100)
model.save('auto_encoder_text2.h5')

model.fit(train_X, train_Y, epochs=4,
			verbose=1,
			validation_data = (test_X,test_Y))
model.save('auto_encoder_text2.h5')
'''