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

import pickle
import os
from os import listdir
import cv2
import numpy as np
import random
from tqdm import tqdm
from glob import glob
import csv
from skimage.transform import resize
from PIL import ImageFile,Image

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import VGG19,VGG16,InceptionV3,ResNet50,DenseNet121
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


random.seed(42)
def get_list_of_concepts(data):
	desc = list()
	for key in data:
		concepts = data[key]
		concepts = ' '.join(concepts)
		#edit to fit 
		desc.append('startseq '+concepts+' endseq')
	return desc

def get_list_of_concepts_(data):
	desc = list()
	for data_object in data:
		concepts = data_object['concepts']
		concepts = ' '.join(concepts)
		desc.append('startseq '+concepts+' endseq')
	return desc

def data_from_concept_csvs(csv_paths):
	data_dict = dict()
	for path in csv_paths:
		with open(path, newline='') as csvfile:
			reader = csv.reader(csvfile,delimiter=' ', quotechar='|')
			for row in reader:
				row_data = row[0].split(',')
				filename = row_data[0]
				concepts = row_data[1:]
				data_dict[filename] = concepts
	return data_dict

def create_tokenizer(descriptions):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(descriptions)
	return tokenizer

def load_data(data_file):
	data = pickle.load(open(data_file, 'rb'))
	return data

def binary_encoding(vocab_size,index_list):
	empty_arr = np.zeros(vocab_size)
	for arr_index in index_list:
		empty_arr[arr_index]=1

	return empty_arr

def text_feature(model, vocab_size,concepts,tokenizer):

	seq = tokenizer.texts_to_sequences([concepts])[0]

	in_seq = seq[:]	
	in_seq = pad_sequences([in_seq], maxlen=142)[0]
	in_seq = np.expand_dims(in_seq,axis=0)
	#seq = seq.reshape(seq)
	#encoded = binary_encoding(vocab_size,seq)
	#comment next line out for shape errors
	#encoded = encoded.reshape(1,encoded.shape[0],1)
	feature = model.predict(in_seq)

	return feature

def image_feature(path,model):
	try:
		image = load_img(path, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	except:
		image = cv2.imread(path)
		image =  cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
		image = Image.fromarray(image)
		image = img_to_array(image)
	image = image.astype("float32")/255.0
	image = np.expand_dims(image,axis=0)
	
	feature = model.predict(image)
	feature = feature.flatten()

	return feature

def extract(dataset=None,concept_csv_paths=None,reduce_by=0,model=None,output_layer=-1,
			text_extract=None,text_model=None,output_layer_text=-1):
	concept_data = data_from_concept_csvs(concept_csv_paths)
	concept_list = get_list_of_concepts(concept_data)
	if model in ['VGG16','VGG19','RESNET50',
			'INCEPTION','DENSE121','XCEPTION'] and (text_extract != 'only' or text_extract == None):
			if model == 'VGG19':
				model = VGG19(weights='imagenet')
				model = Model(inputs=model.inputs, 
							outputs=model.layers[output_layer].output)
			elif model == 'VGG16':
				model = VGG16(weights='imagenet')
				model = Model(inputs=model.inputs, 
							outputs=model.layers[output_layer].output)
			elif model == 'RESNET50':
				model = ResNet50(weights='imagenet')
				model = Model(inputs=model.inputs, 
							outputs=model.layers[output_layer].output)
			elif model == 'DENSE121':
				model = DenseNet121(weights='imagenet')
				model = Model(inputs=model.inputs, 
							outputs=model.layers[output_layer].output)
			elif model == 'INCEPTION':
				model = InceptionV3(weights='imagenet')
				model = Model(inputs=model.inputs, 
							outputs=model.layers[output_layer].output)
			elif model == 'XCEPTION':
				model = InceptionV3(weights='imagenet')
				model = Model(inputs=model.inputs, 
							outputs=model.layers[output_layer].output)
			print(model.summary())

	elif model not in ['VGG16','VGG19','RESNET50',
			'INCEPTION','DENSE121','XCEPTION'] and (text_extract != 'only' or text_extract == None):
		model = load_model(model)
		model = Model(inputs=model.inputs, 
							outputs=model.layers[output_layer].output)
		print(model.summary())

	if text_extract == 'only' or text_extract == 'with':
		model2 = load_model(text_model)
		model2 = Model(inputs=model2.inputs, 
					outputs=model2.get_layer(output_layer_text).output)
		print(model2.summary())
		
	tokenizer = create_tokenizer(concept_list)
	vocab_size = len(tokenizer.word_index) + 1

	
	
	PATHS = [path for path in glob(dataset+'/*/*/*')]
	random.shuffle(PATHS)

	if reduce_by is not None:
		start = int(len(PATHS)*reduce_by)
		PATHS = PATHS[start:]

	PATHS = tqdm(PATHS)
	PATHS.set_description("Extracting Features")

	data_list = list()

	for path in PATHS:

		data_object = dict()

		path_split = os.path.split(path)
		image_class = os.path.split(path_split[-2])[-1]
		image_name = path_split[-1]
		concepts = concept_data[image_name]

		data_object['filepath'] = path
		data_object['image_class'] = image_class
		data_object['concepts'] = concepts

		if path is None:
			raise('stop')

		if text_extract != 'only' or text_extract == None:
			data_object['image_feature'] = image_feature(path,model)

		elif text_extract == 'only' or text_extract == 'with':
			data_object['concepts_feature'] = text_feature(model2,vocab_size,concepts,tokenizer)

		data_list.append(data_object)
	
	return [data_list,concept_list,tokenizer,vocab_size]