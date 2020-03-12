from utils import *
from pretext_task_generation import *

data_folder = Path("data/subj")
input_train_txt_path = data_folder.joinpath("train.txt")
input_test_txt_path = data_folder.joinpath("test.txt")
output_folder = data_folder.joinpath("ssl")

num_positions = 2
num_permutations = 4

output_train_txt_path = output_folder.joinpath(f"{num_positions}positions_{num_permutations}permutations_train.txt")
output_swap_examples(input_train_txt_path, output_train_txt_path, num_positions, num_permutations)

output_test_txt_path = output_folder.joinpath(f"{num_positions}positions_{num_permutations}permutations_test.txt")
output_swap_examples(input_test_txt_path, output_test_txt_path, num_positions, num_permutations)