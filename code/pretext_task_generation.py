import random
random.seed(42)
from utils import *

#############################################
########### masked language model ###########
#############################################

def generate_mlm_lines(line, word2idx):
    words = line.split(' ')
    mlm_lines = []
    for i, word in enumerate(words):
        if word in word2idx:
            list_copy = list(words)
            del list_copy[i] #list_copy[i] = ">"
            mlm_data_line = " ".join(list_copy)
            mlm_line = '\t'.join([str(word2idx[word]), mlm_data_line])
            mlm_lines.append(mlm_line)
    return mlm_lines

def output_mlm_examples(input_txt_path, output_txt_path, word2idx_path):

    lines = input_txt_path.open('r').readlines()
    output_writer = output_txt_path.open('w')
    word2idx = load_pickle(word2idx_path)

    counter = 0
    for line in lines:
        line_data = line.replace("\n", "").split('\t')[1]
        mlm_lines = generate_mlm_lines(line_data, word2idx)
        counter += len(mlm_lines)
        for mlm_line in mlm_lines:
            output_writer.write(mlm_line + '\n')
    
    print(f"{counter} mlm lines outputted for {len(lines)}")


#############################################
############## word swap task ###############
#############################################

def retrieve_permutations(num_positions, num_permutations, max_len=15):

    permutations_file = open(f"permutations/{num_positions}n_swap_tokens_{max_len}max_len.txt", 'r').readlines()
    permutation_list = []
    for line in permutations_file:
        big_parts = line[:-1].split(';')
        original_idxes = tuple([int(x) for x in big_parts[0].split(',')])
        new_idxes = tuple([int(x) for x in big_parts[1].split(',')])
        tuple_list = [original_idxes, new_idxes]
        permutation_list.append(tuple_list)
    return [()] + permutation_list[:num_permutations-1]

def generate_swap_examples(token_list, permutations):
    
    x, y = [], []

    hits, misses = 0, 0
    for _class, permutation in enumerate(permutations):
        new_token_list = list(token_list)
        for swap in permutation:
            source = swap[0]
            dest = swap[1]
            if source < len(new_token_list) and dest < len(token_list):
                new_token_list[source] = token_list[dest]
                hits += 1
            else:
                misses += 1
                break
        x.append(new_token_list)
        y.append(_class)
    return x, y, hits, misses

def output_swap_examples(input_txt_path, output_txt_path, num_positions, num_permutations):

    permutations = retrieve_permutations(num_positions, num_permutations)

    lines = input_txt_path.open('r').readlines()
    output_writer = output_txt_path.open('w')
    
    total_hits, total_misses = 0, 0
    for line in lines:
        data = line.replace("\n", "").split('\t')[1]
        token_list = [x for x in data.split(' ') if x != ""]
        x_sentence, y_sentence, hits, misses = generate_swap_examples(token_list, permutations)
        for x_sentence_i, y_sentence_i in zip(x_sentence, y_sentence):
            line = str(y_sentence_i) + '\t' + ' '.join(x_sentence_i) + '\n'
            output_writer.write(line)
        total_hits += hits; total_misses += misses
    
    print(f"{total_hits} hits and {total_misses} misses")


#############################################
############## insertion task ###############
#############################################

def retrieve_insertion_permutations(num_insertions, num_permutations, max_len=15):

    permutations_file = open(f"permutations/{num_insertions}n_insertions_{max_len}max_len.txt", 'r').readlines()
    permutation_list = []
    for line in permutations_file:
        parts = line[:-1].split(';')
        idxes = tuple([int(x) for x in parts[0].split(',')])
        permutation_list.append(idxes)
    return permutation_list[:num_permutations]

def generate_insertion_examples(token_list, permutations, middle_word_list):

    insertion_lines = []
    insertion_classes = []
    for _class, insertion_idxes in enumerate(permutations):
        token_list_copy = list(token_list)
        for insertion_idx in insertion_idxes:
            if insertion_idx < len(token_list_copy) and len(token_list_copy[insertion_idx]) >= 1:
                random_word = random.choice(middle_word_list)
                token_list_copy[insertion_idx] += ' ' + random_word
        insertion_line = ' '.join(token_list_copy)
        insertion_lines.append(insertion_line)
        insertion_classes.append(_class)
    return insertion_lines, insertion_classes

def output_insertion_examples(input_txt_path, output_txt_path, middle_word_list_path, num_insertions, num_permutations):

    permutations = retrieve_insertion_permutations(num_insertions, num_permutations)
    middle_word_list = load_pickle(middle_word_list_path)
    print(permutations)

    lines = input_txt_path.open('r').readlines()
    output_writer = output_txt_path.open('w')

    counter = 0
    for line in lines:
        data_line = line.replace("\n", "").split('\t')[1]
        token_list = [x for x in data_line.split(' ') if x != ""]
        insertion_lines, insertion_classes = generate_insertion_examples(token_list, permutations, middle_word_list)
        for insertion_line, insertion_class in zip(insertion_lines, insertion_classes):
            output_writer.write('\t'.join([str(insertion_class), insertion_line + '\n']))
            counter += 1
    
    print(f"{counter} insertion examples generated from {len(lines)} lines")