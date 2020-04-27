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

import sys
sys.path.append("..")

import tensorflow as tf

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from LSTM_model_train import LS
from models import  AlternativeRNNModel
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from extract import extract

DATASET = os.path.join('C:/Users/mdrah/Thesis','Concept_Train_2020')

CSV_PATHS = [path for path in glob('C:/Users/mdrah/Thesis/a1f6f33d-e7fa-400b-8b9e-5c559212d2ae_ImageCLEF2020_Concept-Detection_Training-Set_Concepts/*')]

'''
model = LS(DATASET,CSV_PATHS,None,0.1,'VGG16',-3,'with',
	'C:/Users/mdrah/New_Thesis/autoencoder_text-ep002-loss0.003-val_loss0.003.h5'
	,'encoded')
'''
EPOCHS = 50
INIT_LR = 1e-3
OPT = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
BATCH_SIZE=5
VERBOSE=1

filepath = 'dense_121-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
'''			
			dataset=None,concept_csv_paths=None,features_path=None,
			reduce_by=0,model=None,output_layer=None,
			text_extract=None,text_model=None,output_layer_text=None,
			LSTM=None,epochs=None,opt=None,use_generator=False,
			batch_size=None, verbose = 1, checkpoint = None
'''
model = LS(dataset=DATASET,concept_csv_paths=CSV_PATHS,
			features_path='dense_121_feat.pkl',reduce_by=0.5,
	 LSTM=AlternativeRNNModel,epochs=EPOCHS,opt=OPT,
	 use_generator=True,batch_size=BATCH_SIZE,verbose=VERBOSE,
	 checkpoint=checkpoint)
'''
features = extract(dataset=DATASET,concept_csv_paths=CSV_PATHS,model='DENSE121',output_layer=-2)

with open('dense_121_feat.pkl',"wb") as pickle_input:
	pickle.dump(features,pickle_input)
'''