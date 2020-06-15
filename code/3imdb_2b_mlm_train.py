from utils import *
from utils_model import *
import config as config

dataset_name = "imdb"
data_folder = config.data_folders[dataset_name]
mlm_folder = data_folder.joinpath("mlm")
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)

train_path = mlm_folder.joinpath("mlm_train.txt")
test_path = mlm_folder.joinpath("mlm_test.txt")
checkpoints_folder = Path("checkpoints").joinpath(dataset_name).joinpath(f"mlm")
checkpoints_folder.mkdir(parents=True, exist_ok=True)

train_ssl(train_path, test_path, 1000, word2vec, checkpoints_folder)