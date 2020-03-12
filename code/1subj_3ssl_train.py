from utils import *
from utils_model import *
import config as config

dataset_name = "subj"
data_folder = config.data_folders[dataset_name]
ssl_folder = data_folder.joinpath("ssl")
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)

num_positions = 2
num_permutations = 2
train_path = ssl_folder.joinpath(f"{num_positions}positions_{num_permutations}permutations_train.txt")
test_path = ssl_folder.joinpath(f"{num_positions}positions_{num_permutations}permutations_test.txt")
checkpoints_folder = Path("checkpoints").joinpath(dataset_name).joinpath(f"{num_positions}positions_{num_permutations}permutations")
checkpoints_folder.mkdir(parents=True, exist_ok=True)

train_ssl(train_path, test_path, num_permutations, word2vec, checkpoints_folder)