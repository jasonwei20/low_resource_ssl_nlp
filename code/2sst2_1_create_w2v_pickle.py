import config as config
from utils import *

dataset_name = "sst2"
data_folder = config.data_folders[dataset_name]
output_pickle_path = config.word2vec_folder.joinpath(dataset_name + "_w2v.pkl")
output_word2idx_path = config.word2vec_folder.joinpath(dataset_name + "_word2idx.pkl")
output_middle_words_path = config.word2vec_folder.joinpath(dataset_name + "_middle_words.pkl")

gen_vocab_dicts(data_folder, output_pickle_path, output_word2idx_path, output_middle_words_path, config.word2vec_path)