from utils import *
import config as config

data_folder = Path("data/subj")
output_pickle_path = Path("word2vec").joinpath("subj_w2v.pkl")

gen_vocab_dicts(data_folder, output_pickle_path, config.word2vec_path)