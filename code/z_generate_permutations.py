from itertools import combinations
from pathlib import Path
import random
random.seed(42)

def generate_insertion_permutations(n_insertions, max_len, max_permutations = 1024):

    token_idxes = range(max_len)
    insert_positions_list = list(combinations(token_idxes, n_insertions))
    random.shuffle(insert_positions_list)
    
    output_writer = Path(f"permutations/{n_insertions}n_insertions_{max_len}max_len.txt").open("w")

    for insert_positions in insert_positions_list[:max_permutations]:
        output_line = ','.join([str(x) for x in insert_positions])
        output_writer.write(output_line+'\n')

#############################################
############## word swap task ###############
#############################################

def is_complete_swap(l_1, l_2):
    for x_1, x_2 in zip(l_1, l_2):
        if x_1 == x_2:
            return False
    return True

def generate_swap(l):
    l_list = list(l)
    candidate = list(l_list)
    random.shuffle(candidate)
    while not is_complete_swap(l_list, candidate):
        random.shuffle(candidate)
    return tuple(candidate)

def generate_permutations(n_swap_tokens, max_len, max_permutations = 1024):

    token_idxes = range(max_len)
    swap_tokens_list = list(combinations(token_idxes, n_swap_tokens))
    random.shuffle(swap_tokens_list)
    
    output_writer = Path(f"permutations/{n_swap_tokens}n_swap_tokens_{max_len}max_len.txt").open("w")

    for swap_tokens in swap_tokens_list[:max_permutations]:
        new_token_order = generate_swap(swap_tokens)
        output_line = ";".join([",".join([str(x) for x in swap_tokens]), ",".join([str(x) for x in new_token_order])])
        output_writer.write(output_line+'\n')

if __name__ == "__main__":

    # for n_swap_tokens in [2, 3, 4, 5]:
    #     generate_permutations(n_swap_tokens=n_swap_tokens, max_len=10)

    for n_insertions in [1, 2, 3, 4, 5]:
        generate_insertion_permutations(n_insertions=n_insertions, max_len=15)
