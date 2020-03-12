


def generate_permutations(num_positions, num_permutations):

    if num_positions == 2 and num_permutations == 2:
        return [[], [(3, 6), (6, 3)]]
    
    return None

def generate_swap_examples(token_list, num_positions, num_permutations):

    permutations = generate_permutations(num_positions, num_permutations)
    
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
                x.append(new_token_list)
                y.append(_class)
            else:
                misses += 1
    return x, y, hits, misses

def output_swap_examples(input_txt_path, output_txt_path, num_positions, num_permutations):

    lines = input_txt_path.open('r').readlines()
    output_writer = output_txt_path.open('w')
    
    total_hits, total_misses = 0, 0
    for line in lines:
        data = line.replace("\n", "").split('\t')[1]
        token_list = [x for x in data.split(' ') if x != ""]
        x_sentence, y_sentence, hits, misses = generate_swap_examples(token_list, num_positions, num_permutations)
        for x_sentence_i, y_sentence_i in zip(x_sentence, y_sentence):
            line = str(y_sentence_i) + '\t' + ' '.join(x_sentence_i) + '\n'
            output_writer.write(line)
        total_hits += hits; total_misses += misses
    
    print(f"{total_hits} hits and {total_misses} misses")