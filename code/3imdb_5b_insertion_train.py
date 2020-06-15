import config as config
from utils import *
from utils_model import *

dataset_name = "imdb"
data_folder = config.data_folders[dataset_name]
ssl_folder = data_folder.joinpath("ssl")
word2vec_pickle = f"word2vec/{dataset_name}_w2v.pkl"
word2vec = load_pickle(word2vec_pickle)

for num_insertions in [1, 2, 3, 4, 5]:

    for num_permutations in [2, 8]:

        print(f"num_insertions={num_insertions}, num_permutations={num_permutations}")
        train_path = ssl_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_train.txt")
        test_path = ssl_folder.joinpath(f"{num_insertions}insertions_{num_permutations}permutations_test.txt")
        checkpoints_folder = Path("checkpoints").joinpath(dataset_name).joinpath(f"{num_insertions}insertions_{num_permutations}permutations")
        checkpoints_folder.mkdir(parents=True, exist_ok=True)

        train_ssl(train_path, test_path, num_permutations, word2vec, checkpoints_folder)
