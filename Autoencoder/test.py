import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFile,Image
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
#import numpy as np
import pickle
from glob import glob

import sys
import numpy as np
import pickle
from tqdm import tqdm
sys.path.append("..")

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from extract import binary_encoding
from models import auto_encoder_text2

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

ImageFile.LOAD_TRUNCATED_IMAGES = True 

text_model = load_model('auto_encoder_text2_.h5')
print(text_model.summary())
text_encoder = Model(inputs=text_model.input,
	outputs=text_model.get_layer("encoded").output)
'''
decoder_input = Input(shape=(256,))
next_input = decoder_input

for layer in text_model.layers[-3:]:
    next_input = layer(next_input)

text_decoder = Model(inputs=decoder_input,
	outputs=next_input)
print(text_decoder.summary())
'''

def create_sequences(tokenizer, max_length, vocab_size, 
	data):
	x, y = list(), list()
	# walk through each data object
	
	data = tqdm(data)
	data.set_description("Creating Sequences")

	for data_object in data:
		
		image_feature = data_object['image_feature']
		concepts = data_object['concepts']
		concepts = ' '.join(concepts)
		#concepts = 'startseq '+concepts+' endseq'
		seq = tokenizer.texts_to_sequences([concepts])[0]
		
		in_seq = seq[:]
		out_seq = in_seq
		
		in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
		
		#out_seq = binary_encoding(vocab_size,[out_seq])
		out_seq = to_categorical([in_seq], num_classes=vocab_size)[0]
		#out_seq = np.expand_dims(out_seq)
		#print(out_seq.shape)
		x.append(in_seq)
		y.append(out_seq)
	
	return np.array(x), np.array(y)


def load_data(data_file):
	data = pickle.load(open(data_file, 'rb'))
	return data


data_list,concept_list,tokenizer,vocab_size = load_data('../RNN_concept_model/dense_121_feat.pkl')

dataset_number = len(data_list)

test_data = data_list[int(0.95*dataset_number):]
test_X,test_Y = create_sequences(tokenizer, 142, vocab_size, 
	test_data)
print(test_X[1])

b = test_X[1].reshape(1,test_X[1].shape[0])
a = text_model.predict(b)
a = a[0]>0.5
a = a.astype('int')
#print(list(a.astype('int')))
#print(list(test_X[0]))

out_seq = binary_encoding(vocab_size,[test_X[1]])
out_seq = out_seq.astype('int')
c = a*out_seq
#print(list(c))
count = 0
for el in c:
	if el == 1:
		count+=1
print(list(c))
print(count)