import torch
from transformers import *
import numpy as np

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'
model = model_class.from_pretrained(pretrained_weights)
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

def get_bert_embedding_single(model, tokenizer, input_text):
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    last_hidden_states = model(input_ids)[0].cpu().detach().numpy()
    last_hidden_states = np.mean(last_hidden_states, axis=1)
    last_hidden_states = last_hidden_states.flatten()
    return last_hidden_states

def get_bert_embedding(model, tokenizer, input_text_list, embedding_size=768):
    extracted_features_list = np.zeros((len(input_text_list), embedding_size))
    for i, input_text in enumerate(input_text_list):
        extracted_features = get_bert_embedding_single(model, tokenizer, input_text)
        extracted_features_list[i, :] = extracted_features
    return extracted_features_list

def get_sentence_list(txt_path):
    lines = open(txt_path, 'r').readlines()
    lines = [x.split('\t')[-1][:-1] for x in lines]
    return lines

# extracted_features_list = get_bert_embedding(model, tokenizer, ['hello i am jason', 'i like machine learning'])

import config as config
from utils import *
from utils_model import *

dataset_name = "sst2"
data_folder = config.data_folders[dataset_name]
num_classes = config.num_classes_dict[dataset_name]
ssl_folder = data_folder.joinpath("ssl")
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)

train_txt_path = data_folder.joinpath("train.txt")
test_txt_path = data_folder.joinpath("test.txt")

train_lines = get_sentence_list(train_txt_path)
test_lines = get_sentence_list(test_txt_path)

train_extracted_features = get_bert_embedding(model, tokenizer, train_lines)
test_extracted_features = get_bert_embedding(model, tokenizer, test_lines)

train_x, train_y = get_x_y(train_txt_path, num_classes, word2vec_len=300, input_size=40, word2vec=word2vec)
test_x, test_y = get_x_y(test_txt_path, num_classes, word2vec_len=300, input_size=40, word2vec=word2vec)

k_per_class_to_n_voters = {	1: 1,
                            2: 1,
                            3: 1, 
                            5: 3,
                            10: 3,
                            20: 5}

for k_per_class, n_voters in k_per_class_to_n_voters.items():
    calculate_few_shot_acc(train_extracted_features, train_y, test_extracted_features, test_y, num_classes, k_per_class, n_voters)
	