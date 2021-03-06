import config as config
from utils import *
from utils_model import *

dataset_name = "subj"
data_folder = config.data_folders[dataset_name]
num_classes = config.num_classes_dict[dataset_name]
ssl_folder = data_folder.joinpath("ssl")
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)

train_txt_path = data_folder.joinpath("train.txt")
test_txt_path = data_folder.joinpath("test.txt")

checkpoint_paths = [Path("checkpoints/subj/mlm_first_run/cnn_e26_tacc0.0997_vacc0.0957.pt"),
                    Path("checkpoints/subj/supervised/cnn_e18_tacc0.8792_vacc0.8760.pt"),
                    Path("checkpoints/subj/2positions_8permutations/cnn_e07_tacc0.2003_vacc0.1811.pt"),
                    Path("checkpoints/subj/2insertions_4permutations/cnn_e14_tacc0.4704_vacc0.3713.pt"),]

for checkpoint_path in checkpoint_paths:
    evaluate_ssl_model(train_txt_path, test_txt_path, num_classes, word2vec, checkpoint_path)