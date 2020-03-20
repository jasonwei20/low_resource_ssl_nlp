import config as config
from utils import *
from utils_model import *

dataset_name = "sst2"
data_folder = config.data_folders[dataset_name]
num_classes = config.num_classes_dict[dataset_name]
ssl_folder = data_folder.joinpath("ssl")
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)

train_txt_path = data_folder.joinpath("train.txt")
test_txt_path = data_folder.joinpath("test.txt")

checkpoint_paths = [Path("checkpoints/sst2/mlm/cnn_e00_tacc0.0031_vacc0.0056.pt"),
                    Path("checkpoints/sst2/mlm/cnn_e01_tacc0.0329_vacc0.0594.pt"),
                    Path("checkpoints/sst2/mlm/cnn_e02_tacc0.0663_vacc0.0623.pt"),
                    Path("checkpoints/sst2/mlm/cnn_e03_tacc0.0707_vacc0.0773.pt"),
                    Path("checkpoints/sst2/mlm/cnn_e10_tacc0.0855_vacc0.0835.pt"),
                    Path("checkpoints/sst2/mlm/cnn_e28_tacc0.1027_vacc0.0904.pt"),
                    Path("checkpoints/sst2/mlm/cnn_e03_tacc0.0707_vacc0.0773.pt"),]

for checkpoint_path in checkpoint_paths:
    evaluate_ssl_model(train_txt_path, test_txt_path, num_classes, word2vec, checkpoint_path)