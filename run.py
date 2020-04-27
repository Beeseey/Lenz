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
import os
from glob import glob
import pickle
import numpy as np

import tensorflow as tf

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from Autoencoder.train import train
from Autoencoder.models import auto_encoder,auto_encoder_text 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from extract import extract

from Retrieval.search import restructure, train_tree, get_results


def load_data(data_file):
	data = pickle.load(open(data_file, 'rb'))
	return data
'''
data_list,concept_list,tokenizer,vocab_size = load_data('RNN_concept_model/dense_121_feat.pkl')
dataset_number = len(data_list)
train_data = data_list[:int(0.85*dataset_number)]
test_data = data_list[int(0.85*dataset_number):]

EPOCHS = 10
INIT_LR = 1e-3
OPT = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
BATCH_SIZE=100
VERBOSE=1

image_autoencoder = auto_encoder(width=224, height=224, depth=3, filters=(32,64), 
	latentDim=256)
print(image_autoencoder.summary())
#text_autoencoder = auto_encoder_text(maxlen=142,vocab_size=3050,embedding_size=256,
#	latent_dim=256)

#raise ('stop')

filepath = 'Autoencoder-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model = train(autoencoder=image_autoencoder,text_or_image='image',train_data=train_data,test_data=test_data,
	EPOCHS=EPOCHS,OPT=OPT,use_generator=True,BATCH_SIZE=BATCH_SIZE,
	VERBOSE=VERBOSE,Checkpoint=checkpoint,tokenizer=tokenizer,max_length=142,
	vocab_size=vocab_size,image_input_size=(224,224))
'''
#dataset=None,concept_csv_paths=None,reduce_by=0,model=None,output_layer=-1,
#			text_extract=None,text_model=None,output_layer_text=-1
#features = extract(dataset=DATASET,concept_csv_paths=CSV_PATHS,model='DENSE121',output_layer=-2)
'''
DATASET = 'C:/Users/mdrah/Thesis/Concept_Train_2020'
CSV_PATHS = [path for path in glob('C:/Users/mdrah/Thesis/a1f6f33d-e7fa-400b-8b9e-5c559212d2ae_ImageCLEF2020_Concept-Detection_Training-Set_Concepts/*')]
text_model = 'Autoencoder/auto_encoder_text2_.h5'

features = extract(dataset = DATASET,concept_csv_paths=CSV_PATHS,text_extract='only',
					text_model=text_model, output_layer_text='encoded')

with open('text_feat.pkl',"wb") as pickle_input:
	pickle.dump(features,pickle_input)
'''

def MSE(x,y):
	MSE = np.square(np.subtract(x,y)).mean()

	return MSE

'''

#data_list,concept_list,tokenizer,vocab_size = load_data('RNN_concept_model/dense_121_feat.pkl')
data_list,concept_list,tokenizer,vocab_size = load_data('text_feat.pkl')

feat_list, data_dict = restructure(data_list, 'text')

tree = train_tree(feat_list,MSE)

query = feat_list[10]
b = data_dict[tuple(query)]
#print(b)

results,scores = get_results(data_dict,query,tree,5,'text')

print(results)
'''