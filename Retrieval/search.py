from numpy import dot
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm

import pickle
import vptree

from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
sys.setrecursionlimit(20000)

def load_data(data_file):
	data = pickle.load(open(data_file, 'rb'))
	return data

def restructure(data_list, image_or_text):

	if image_or_text == 'image':
		data_dict = {tuple(data_object['image_feature']):data_object['filepath'] for data_object in data_list}
		feat_list = [data_object['image_feature'] for data_object in data_list]
	elif image_or_text == 'text':
		data_dict = {}
		feat_list = []
		data_list = tqdm(data_list)
		data_list.set_description("Restructuring")
		for data_object in data_list:
			concepts = data_object['concepts']
			in_seq = data_object['concepts_feature'][0]
			if tuple(in_seq) not in feat_list:
				data_dict[tuple(in_seq)] = [[data_object['filepath'],data_object['concepts']]]
				feat_list.append(tuple(in_seq))
			else:
				data_dict[tuple(in_seq)].append([data_object['filepath'],
										data_object['concepts']])

	return feat_list, data_dict 

def train_tree(features_list,dist_measure):
	tree = vptree.VPTree(features_list,dist_measure)
	return tree

def get_results(data_dict,query,tree,query_no,image_or_text):

	search_results = []
	scores = []
	results = tree.get_n_nearest_neighbors(query, query_no)
	if image_or_text == 'image':
		for key,value in results:
			value = tuple(value)
			search_results.append(data_dict[value])
			scores.append(key)
	elif image_or_text == 'text':
		count = 0
		for key,value in results:
			value = tuple(value)
			if len(data_dict[value]) > 1:
				for result in data_dict[value]:
					search_results.append(result)
					scores.append(key)
					count+=1
					if count == 10:
						break
			else:
				search_results.append(data_dict[value][0])
				count+=1
				if count == 10:
					break
	return search_results,scores