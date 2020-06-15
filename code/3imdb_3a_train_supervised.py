import config as config
from utils import *
from utils_model import *

dataset_name = "imdb"
data_folder = config.data_folders[dataset_name]
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)

train_path = data_folder.joinpath(f"train.txt")
test_path = data_folder.joinpath(f"test.txt")
checkpoints_folder = Path("checkpoints").joinpath(dataset_name).joinpath(f"supervised")
checkpoints_folder.mkdir(parents=True, exist_ok=True)

num_classes = config.num_classes_dict[dataset_name]

train_ssl(train_path, test_path, num_classes, word2vec, checkpoints_folder)