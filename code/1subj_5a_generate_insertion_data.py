import config as config
from pretext_task_generation import *
from utils import *

dataset_name = "subj"
data_folder = config.data_folders[dataset_name]
input_train_txt_path = data_folder.joinpath("train.txt")
input_test_txt_path = data_folder.joinpath("test.txt")
output_folder = data_folder.joinpath("ssl")
middle_word_list_path = config.word2vec_folder.joinpath(dataset_name + "_middle_words.pkl")

for num_insertions in [1, 2, 4]:
    for num_permutations in [2, 4, 8, 16, 32]:
        
        output_train_txt_path = output_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_train.txt")
        output_insertion_examples(input_train_txt_path, output_train_txt_path, middle_word_list_path, num_insertions, num_permutations)

        output_test_txt_path = output_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_test.txt")
        output_insertion_examples(input_test_txt_path, output_test_txt_path, middle_word_list_path, num_insertions, num_permutations)