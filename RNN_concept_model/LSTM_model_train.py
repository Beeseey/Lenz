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


import sys
sys.path.append("..")

from extract import extract, get_list_of_concepts_, create_tokenizer
from train import train

def load_data(data_file):
	data = pickle.load(open(data_file, 'rb'))
	return data

def max_length(descriptions):
	return max(len(d.split()) for d in descriptions)

def LS(dataset=None,concept_csv_paths=None,features_path=None,
			reduce_by=0,model=None,output_layer=None,
			text_extract=None,text_model=None,output_layer_text=None,
			LSTM=None,epochs=None,opt=None,use_generator=False,
			batch_size=None, verbose = 1, checkpoint = None):
	if features_path is None:
		data_list,concept_list,tokenizer,vocab_size= extract(dataset,concept_csv_paths,reduce_by,model,output_layer,
							text_extract,text_model,output_layer_text)
	else:
		data_list,concept_list,tokenizer,vocab_size = load_data(features_path)
		if reduce_by is not 0:
			start = int(len(data_list)*reduce_by)
			data_list= data_list[start:]
			concept_list = get_list_of_concepts_(data_list)

			tokenizer = create_tokenizer(concept_list)
			vocab_size = len(tokenizer.word_index) + 1

		print(len(data_list))

	#max_length = max_length(concept_list)
	dataset_number = len(data_list)
	train_data = data_list[:int(0.98*dataset_number)]
	test_data = data_list[int(0.98*dataset_number):]

	LSTM = LSTM(1024,vocab_size,142)

	model = train(LSTM=LSTM,train_data=train_data,test_data=test_data,
	EPOCHS=epochs,OPT=opt,use_generator=use_generator,BATCH_SIZE=batch_size,
	VERBOSE=verbose,Checkpoint=checkpoint,tokenizer=tokenizer,
	max_length=142,vocab_size=vocab_size)
