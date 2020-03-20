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

checkpoint_paths = [Path("checkpoints/sst2/4insertions_8permutations/cnn_e00_tacc0.1314_vacc0.1364.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e03_tacc0.1784_vacc0.1630.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e07_tacc0.2323_vacc0.1986.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e11_tacc0.2816_vacc0.2298.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e15_tacc0.3241_vacc0.2509.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e19_tacc0.3631_vacc0.2607.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e25_tacc0.4168_vacc0.2704.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e59_tacc0.6067_vacc0.2804.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e63_tacc0.6199_vacc0.2812.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e76_tacc0.6635_vacc0.2822.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e77_tacc0.6671_vacc0.2829.pt"),
                    Path("checkpoints/sst2/4insertions_8permutations/cnn_e78_tacc0.6700_vacc0.2834.pt"),
                    Path("checkpoints/sst2/1insertions_8permutations/cnn_e00_tacc0.1266_vacc0.1306.pt"),
                    Path("checkpoints/sst2/1insertions_8permutations/cnn_e08_tacc0.1717_vacc0.1527.pt"),
                    Path("checkpoints/sst2/1insertions_8permutations/cnn_e15_tacc0.2151_vacc0.1700.pt"),
                    Path("checkpoints/sst2/1insertions_8permutations/cnn_e19_tacc0.2417_vacc0.1740.pt"),
                    Path("checkpoints/sst2/1insertions_8permutations/cnn_e21_tacc0.2574_vacc0.1767.pt"),
                    Path("checkpoints/sst2/1insertions_8permutations/cnn_e23_tacc0.2714_vacc0.1776.pt"),
                    Path("checkpoints/sst2/1insertions_8permutations/cnn_e27_tacc0.3000_vacc0.1788.pt"),
                    ]

for checkpoint_path in checkpoint_paths:
    evaluate_ssl_model(train_txt_path, test_txt_path, num_classes, word2vec, checkpoint_path)
