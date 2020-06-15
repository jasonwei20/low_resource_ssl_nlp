from pathlib import Path

word2vec_len = 300
word2vec_path = Path("/home/brenta/scratch/jason/data/word2vec/glove.42B.300d.txt")
word2vec_folder = Path("word2vec")

data_folders = {"subj": Path("data/subj"),
                "sst2": Path("data/sst2"),
                "imdb": Path("data/imdb"),
                }

num_classes_dict = {"subj": 2,
                    "sst2": 2,
                    "imdb": 2,
                    }