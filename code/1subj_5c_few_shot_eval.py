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

# checkpoint_paths = [Path("checkpoints/subj/1insertions_4permutations/cnn_e04_tacc0.2886_vacc0.2618.pt"),
#                     Path("checkpoints/subj/2insertions_4permutations/cnn_e09_tacc0.4043_vacc0.3315.pt"),
#                     Path("checkpoints/subj/2insertions_4permutations/cnn_e71_tacc0.9317_vacc0.4175.pt"),
#                     Path("checkpoints/subj/4insertions_4permutations/cnn_e50_tacc0.8750_vacc0.4015.pt"),
#                     Path("checkpoints/subj/2insertions_2permutations/cnn_e18_tacc0.7509_vacc0.6275.pt"),
#                     Path("checkpoints/subj/2insertions_4permutations/cnn_e14_tacc0.4704_vacc0.3713.pt"),
#                     Path("checkpoints/subj/2insertions_8permutations/cnn_e15_tacc0.3049_vacc0.2100.pt"),
#                     Path("checkpoints/subj/2insertions_16permutations/cnn_e10_tacc0.1366_vacc0.1017.pt"),
#                     Path("checkpoints/subj/2insertions_32permutations/cnn_e12_tacc0.1028_vacc0.0699.pt"),
#                     ]

# checkpoint_paths = [Path("checkpoints/subj/2positions_8permutations/cnn_e00_tacc0.1250_vacc0.1283.pt"),
#                     Path("checkpoints/subj/2positions_8permutations/cnn_e04_tacc0.1644_vacc0.1511.pt"),
#                     Path("checkpoints/subj/2positions_8permutations/cnn_e07_tacc0.2003_vacc0.1811.pt"),
#                     Path("checkpoints/subj/2positions_8permutations/cnn_e10_tacc0.2351_vacc0.2154.pt"),
#                     Path("checkpoints/subj/2positions_8permutations/cnn_e15_tacc0.2959_vacc0.2465.pt"),
#                     Path("checkpoints/subj/2positions_8permutations/cnn_e46_tacc0.4116_vacc0.2733.pt"),
#                     ]

checkpoint_paths = [Path("checkpoints/subj/2insertions_8permutations/cnn_e00_tacc0.1262_vacc0.1281.pt"),
                    Path("checkpoints/subj/2insertions_8permutations/cnn_e03_tacc0.1639_vacc0.1509.pt"),
                    Path("checkpoints/subj/2insertions_8permutations/cnn_e08_tacc0.2262_vacc0.1820.pt"),
                    Path("checkpoints/subj/2insertions_8permutations/cnn_e15_tacc0.3049_vacc0.2100.pt"),
                    Path("checkpoints/subj/2insertions_8permutations/cnn_e70_tacc0.5573_vacc0.2286.pt"),
                    ]

for checkpoint_path in checkpoint_paths:
    evaluate_ssl_model(train_txt_path, test_txt_path, num_classes, word2vec, checkpoint_path)
