import re 
import os
from datasets import load_dataset


def generate_character_set(lang):
    try:
        dataset = load_dataset("common_voice", lang)
    except Exception as data_error:
        raise Exception(f'Could not load this dataset from Common Voice. See error below: {data_error}')

    train = [d for d in dataset['train']]
    valid = [d for d in dataset['validation']]
    test = [d for d in dataset['test']]

    all_data = train+valid+test
    all_sentences = [d['sentence'].lower().strip() for d in all_data ]
    all_string = ''.join(all_sentences)
    all_string = re.sub(r' ','',all_string)
    chars = list(set(all_string))
    file_path =os.path.join(os.getcwd(),f'{lang}_characters.txt') 
    with open(file_path,'w+',encoding='utf8') as file_:
        for char in chars:
            file_.write(char)
            file_.write('\n')
    print(f'Characters for {lang} have been saved in {file_path} and will be used for current training. \n Please edit this file and add more characters in order to have a more robust model.')
    return file_path 

