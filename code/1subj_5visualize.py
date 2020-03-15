from utils import *
from utils_model import *
import config as config

dataset_name = "subj"
data_folder = config.data_folders[dataset_name]
num_classes = config.num_classes_dict[dataset_name]
ssl_folder = data_folder.joinpath("ssl")
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)

train_txt_path = data_folder.joinpath("train.txt")
test_txt_path = data_folder.joinpath("test.txt")

checkpoint_paths = [Path("checkpoints/subj/2positions_32permutations/cnn_e00_tacc0.0312_vacc0.0323.pt"),
                    Path("checkpoints/subj/2positions_32permutations/cnn_e11_tacc0.0840_vacc0.0683.pt"),]