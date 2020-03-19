import math
import time
import numpy as np
import random
from random import randint
from pathlib import Path
random.seed(3)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #get rid of warnings
from os import listdir
from os.path import isfile, join, isdir
import pickle

###################################################
######### loading folders and txt files ###########
###################################################

def get_now_str():
	return str(strftime("%Y-%m-%d_%H:%M:%S", gmtime()))

def load_pickle(file):
	return pickle.load(open(file, 'rb'))

def get_all_vocab(txt_file_path):
	vocab_words = set()
	lines = txt_file_path.open('r').readlines()
	for line in lines:
		data = line[:-1].split('\t')[1]
		words = data.split(' ')
		for word in words:
			vocab_words.add(word)
	return vocab_words

#get full image paths
def get_txt_paths(folder):
    txt_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and '.txt' in f]
    if join(folder, '.DS_Store') in txt_paths:
        txt_paths.remove(join(folder, '.DS_Store'))
    txt_paths = sorted(txt_paths)
    return txt_paths

#get subfolders
def get_subfolder_paths(folder):
    subfolder_paths = [join(folder, f) for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)]
    if join(folder, '.DS_Store') in subfolder_paths:
        subfolder_paths.remove(join(folder, '.DS_Store'))
    subfolder_paths = sorted(subfolder_paths)
    return subfolder_paths

#get all image paths
def get_all_txt_paths(master_folder):

    all_paths = get_txt_paths(master_folder)
    return all_paths

###################################################
################ data processing ##################
###################################################

def get_most_popular(word_to_count, n_most_popular):
	tuples = word_to_count.items()
	sorted_tuple_list = list(reversed(sorted(tuples, key=lambda x: x[1])))
	most_popular_word_tuples = sorted_tuple_list[:n_most_popular]
	most_popular_words = [x[0] for x in most_popular_word_tuples]
	return most_popular_words

def get_middle_popular(word_to_count, n_most_popular_to_exclude=1000, min_count=3):
	tuples = word_to_count.items()
	sorted_tuple_list = list(reversed(sorted(tuples, key=lambda x: x[1])))
	sorted_tuple_list_min_count = [x for x in sorted_tuple_list if x[1] >= min_count]
	middle_word_tuples = sorted_tuple_list_min_count[n_most_popular_to_exclude:]
	middle_popular_words = [x[0] for x in middle_word_tuples]
	return middle_popular_words

#get the pickle file for the word2vec so you don't have to load the entire huge file each time
def gen_vocab_dicts(folder, output_pickle_path, output_word2idx_path, middle_words_path, huge_word2vec):

	word_to_count = {}
	vocab = set()
	text_embeddings = open(huge_word2vec, 'r').readlines()
	word2vec = {}

	#get all the vocab
	all_txt_paths = get_all_txt_paths(str(folder))
	print(all_txt_paths)

	#loop through each text file
	for txt_path in all_txt_paths:

		# get all the words
		try:
			all_lines = open(txt_path, "r").readlines()
			for line in all_lines:
				data = line[:-1].split('\t')[1]
				words = data.split(' ')
				for word in words:
					vocab.add(word)
					if word in word_to_count:
						word_to_count[word] += 1
					else:
						word_to_count[word] = 1
		except:
			print(txt_path, "has an error")
	
	print(len(vocab), "unique words found")

	# load the word embeddings, and only add the word to the dictionary if we need it
	for line in text_embeddings:
		items = line.split(' ')
		word = items[0]
		if word in vocab:
			vec = items[1:]
			word2vec[word] = np.asarray(vec, dtype = 'float32')
	print(len(word2vec), "matches between unique words and word2vec dictionary")

	pickle.dump(word2vec, open(output_pickle_path, 'wb'))
	print("dictionaries outputted to", output_pickle_path)
	
	# generate word2idx for MLM
	most_popular_words = get_most_popular(word_to_count, n_most_popular=1000)
	word2idx = {}
	for i, word in enumerate(sorted(most_popular_words)):
		word2idx[word] = i
	pickle.dump(word2idx, open(output_word2idx_path, 'wb'))
	print("word2idx outputted to", output_word2idx_path)
	
	# generate middle popular words for insertions
	middle_popular_words = get_middle_popular(word_to_count)
	pickle.dump(middle_popular_words, open(middle_words_path, 'wb'))
	print("middle words outputted to", middle_words_path)
	print(len(middle_popular_words))