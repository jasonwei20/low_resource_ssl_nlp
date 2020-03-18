import config as config
from utils import *
from pretext_task_generation import *

dataset_name = "subj"
data_folder = config.data_folders[dataset_name]
input_train_txt_path = data_folder.joinpath("train.txt")
input_test_txt_path = data_folder.joinpath("test.txt")
output_folder = data_folder.joinpath("mlm")
word2idx_path = config.word2vec_folder.joinpath(dataset_name + "_word2idx.pkl")

output_train_txt_path = output_folder.joinpath("mlm_train.txt")
output_mlm_examples(input_train_txt_path, output_train_txt_path, word2idx_path)

output_test_txt_path = output_folder.joinpath("mlm_test.txt")
output_mlm_examples(input_test_txt_path, output_test_txt_path, word2idx_path)
