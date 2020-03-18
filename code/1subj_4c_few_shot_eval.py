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

# checkpoint_paths = [Path("checkpoints/subj/2positions_8permutations/cnn_e00_tacc0.1262_vacc0.1280.pt"),
#                     Path("checkpoints/subj/2positions_8permutations/cnn_e06_tacc0.1781_vacc0.1685.pt/"),
#                     Path("checkpoints/subj/2positions_8permutations/cnn_e13_tacc0.2516_vacc0.2095.pt/"),
#                     Path("checkpoints/subj/2positions_8permutations/cnn_e65_tacc0.4709_vacc0.2397.pt/"),]

# checkpoint_paths = [Path("checkpoints/subj/2positions_4permutations/cnn_e00_tacc0.2496_vacc0.2595.pt"),
#                     Path("checkpoints/subj/2positions_4permutations/cnn_e11_tacc0.3957_vacc0.3268.pt"),
#                     Path("checkpoints/subj/2positions_4permutations/cnn_e18_tacc0.4797_vacc0.3783.pt"),
#                     Path("checkpoints/subj/2positions_4permutations/cnn_e76_tacc0.8619_vacc0.4067.pt"),]

# checkpoint_paths = [Path("checkpoints/subj/2positions_32permutations/cnn_e00_tacc0.0312_vacc0.0323.pt"),
#                     Path("checkpoints/subj/2positions_32permutations/cnn_e03_tacc0.0436_vacc0.0443.pt"),
#                     Path("checkpoints/subj/2positions_32permutations/cnn_e06_tacc0.0650_vacc0.0609.pt"),
#                     Path("checkpoints/subj/2positions_32permutations/cnn_e11_tacc0.0840_vacc0.0683.pt"),
#                     Path("checkpoints/subj/2positions_32permutations/cnn_e32_tacc0.1203_vacc0.0715.pt"),]

checkpoint_paths = [Path("checkpoints/subj/2positions_2permutations/cnn_e19_tacc0.7041_vacc0.6040.pt"),
                    Path("checkpoints/subj/2positions_4permutations/cnn_e25_tacc0.5806_vacc0.4255.pt"),
                    Path("checkpoints/subj/2positions_8permutations/cnn_e17_tacc0.3118_vacc0.2515.pt"),
                    Path("checkpoints/subj/2positions_16permutations/cnn_e14_tacc0.1695_vacc0.1331.pt"),
                    Path("checkpoints/subj/2positions_32permutations/cnn_e19_tacc0.0991_vacc0.0696.pt"),]

for checkpoint_path in checkpoint_paths:
    evaluate_ssl_model(train_txt_path, test_txt_path, num_classes, word2vec, checkpoint_path)
