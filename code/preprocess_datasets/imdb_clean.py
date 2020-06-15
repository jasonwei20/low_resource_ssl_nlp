import re

from os import listdir
from os.path import isfile, join

def get_file_paths(mypath):
    return['/'.join([mypath, f]) for f in listdir(mypath) if isfile(join(mypath, f))]

#cleaning up text
def get_only_chars(line):

    clean_line = ""

    line = line.lower()
    line = line.replace(" 's", " is") 
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.replace("'", "")

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    # print(clean_line)
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

def parse_data_split(data_split_path, output_path):

    output_writer = open(output_path, 'w')

    _classes = ['pos', 'neg']
    labels = [1, 0]
    for _class, label in zip(_classes, labels):
        my_path = '/'.join([data_split_path, _class])
        file_paths = get_file_paths(my_path)
        for file_path in file_paths:
            text = open(file_path, 'r').readlines()[0]
            only_chars = get_only_chars(text)
            output_writer.write(str(label) + '\t' + only_chars + '\n')


if __name__ == "__main__":

    parse_data_split('aclimdb/train', 'train.txt')
    parse_data_split('aclimdb/test', 'test.txt')