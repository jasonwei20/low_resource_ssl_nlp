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

checkpoint_paths = [Path("checkpoints/subj/mlm/cnn_e00_tacc0.0049_vacc0.0327.pt"),
                    Path("checkpoints/subj/mlm/cnn_e01_tacc0.0573_vacc0.0740.pt"),
                    Path("checkpoints/subj/mlm/cnn_e07_tacc0.0805_vacc0.0778.pt"),
                    Path("checkpoints/subj/mlm/cnn_e12_tacc0.0831_vacc0.0809.pt"),
                    Path("checkpoints/subj/mlm/cnn_e18_tacc0.0902_vacc0.0881.pt"),]

for checkpoint_path in checkpoint_paths:
    evaluate_ssl_model(train_txt_path, test_txt_path, num_classes, word2vec, checkpoint_path)