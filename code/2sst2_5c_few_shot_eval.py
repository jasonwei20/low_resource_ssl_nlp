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

checkpoint_paths = [Path("checkpoints/sst2/1insertions_2permutations/cnn_e43_tacc0.8785_vacc0.5289.pt"),
                    Path("checkpoints/sst2/2insertions_2permutations/cnn_e50_tacc0.9943_vacc0.7553.pt"),
                    Path("checkpoints/sst2/3insertions_2permutations/cnn_e77_tacc0.9850_vacc0.6312.pt"),
                    Path("checkpoints/sst2/4insertions_2permutations/cnn_e73_tacc0.9926_vacc0.6721.pt"),
                    Path("checkpoints/sst2/5insertions_2permutations/cnn_e61_tacc0.9902_vacc0.7421.pt"),
                    Path("checkpoints/sst2/2insertions_16permutations/cnn_e91_tacc0.3155_vacc0.1287.pt"),
                    Path("checkpoints/sst2/3insertions_16permutations/cnn_e47_tacc0.2751_vacc0.1567.pt"),
                    Path("checkpoints/sst2/4insertions_16permutations/cnn_e53_tacc0.2985_vacc0.1622.pt"),
                    Path("checkpoints/sst2/5insertions_16permutations/cnn_e78_tacc0.4084_vacc0.1805.pt"),
                    ]

for checkpoint_path in checkpoint_paths:
    evaluate_ssl_model(train_txt_path, test_txt_path, num_classes, word2vec, checkpoint_path)
