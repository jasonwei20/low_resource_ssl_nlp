import config as config
from utils import *
from utils_model import *

dataset_name = "sst2"
data_folder = config.data_folders[dataset_name]
ssl_folder = data_folder.joinpath("ssl")
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)


# num_insertions = 2
# num_permutations = 16

# print(f"num_insertions={num_insertions}, num_permutations={num_permutations}")
# train_path = ssl_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_train.txt")
# test_path = ssl_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_test.txt")
# checkpoints_folder = Path("checkpoints").joinpath(dataset_name).joinpath(f"{num_insertions}insertions_{num_permutations}permutations_load1insertions_16permutations-cnn_e92_tacc0.2808_vacc0.1420.pt")
# checkpoints_folder.mkdir(parents=True, exist_ok=True)

# train_ssl(train_path, test_path, num_permutations, word2vec, checkpoints_folder, checkpoint_path=Path("checkpoints/sst2/1insertions_16permutations/cnn_e92_tacc0.2808_vacc0.1420.pt"))

# num_insertions = 3
# num_permutations = 16

# print(f"num_insertions={num_insertions}, num_permutations={num_permutations}")
# train_path = ssl_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_train.txt")
# test_path = ssl_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_test.txt")
# checkpoints_folder = Path("checkpoints").joinpath(dataset_name).joinpath(f"{num_insertions}insertions_{num_permutations}permutations_loadload2insertions_16permutations-cnn_e40_tacc0.2172_vacc0.1251.pt")
# checkpoints_folder.mkdir(parents=True, exist_ok=True)

# train_ssl(train_path, test_path, num_permutations, word2vec, checkpoints_folder, checkpoint_path=Path("checkpoints/sst2/2insertions_16permutations_load1insertions_16permutations-cnn_e92_tacc0.2808_vacc0.1420.pt/cnn_e40_tacc0.2172_vacc0.1251.pt"))


num_insertions = 4
num_permutations = 16

print(f"num_insertions={num_insertions}, num_permutations={num_permutations}")
train_path = ssl_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_train.txt")
test_path = ssl_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_test.txt")
checkpoints_folder = Path("checkpoints").joinpath(dataset_name).joinpath(f"{num_insertions}insertions_{num_permutations}permutations_loadloadload3insertions_16permutations-cnn_e41_tacc0.2855_vacc0.1266.pt")
checkpoints_folder.mkdir(parents=True, exist_ok=True)

train_ssl(train_path, test_path, num_permutations, word2vec, checkpoints_folder, checkpoint_path=Path("checkpoints/sst2/3insertions_16permutations_loadload2insertions_16permutations-cnn_e40_tacc0.2172_vacc0.1251.pt/cnn_e41_tacc0.2855_vacc0.1266.pt"))