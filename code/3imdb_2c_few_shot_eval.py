import config as config
from utils import *
from utils_model import *

dataset_name = "imdb"
data_folder = config.data_folders[dataset_name]
num_classes = config.num_classes_dict[dataset_name]
ssl_folder = data_folder.joinpath("ssl")
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)

train_txt_path = data_folder.joinpath("train.txt")
test_txt_path = data_folder.joinpath("test.txt")

checkpoint_paths = [#Path("checkpoints/sst2/mlm_first_run/cnn_e34_tacc0.1122_vacc0.0953.pt"),
                    Path("checkpoints/imdb/mlm/cnn_e72_tacc0.1463_vacc0.1070.pt"),]
                    #Path("checkpoints/sst2/4insertions_8permutations/cnn_e78_tacc0.6700_vacc0.2834.pt"),]

for checkpoint_path in checkpoint_paths:
    evaluate_ssl_model(train_txt_path, test_txt_path, num_classes, word2vec, checkpoint_path)