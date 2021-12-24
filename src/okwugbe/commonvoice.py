import re 
import os
from datasets import load_dataset
import unicodedata
from cvutils import Alphabet

accent_code = [b'\\u0301', b'\\u0300', b'\\u0306', b'\\u0308', b'\\u0303']
alpha = {'ɔ': 0, 'ɛ': 5}
accents = {b'\\u0301': 1, b'\\u0300': 2, b'\\u0306': 3, b'\\u0308': 4, b'\\u0303': 5}
mapping = {1: 'ɔ́', 2: 'ɔ̀', 3: 'ɔ̆', 6: 'έ', 7: 'ὲ', 8: 'ɛ̆'}


def generate_character_set_from_transcripts(lang): #This is no longer in use! 
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
    all_string = unicodedata.normalize("NFC", all_string)
    chars = list(set(all_string))
    return chars

def generate_character_set(lang):
    
    try:
        a = Alphabet(lang.lower().strip())
        characters = list(set(a.get_alphabet())) #This also includes SPACE character
    except Exception:
        print(f'Could not generate alphabets from Common Voice utils for this language -> {lang}. \n Switching to generating character set from raw tokens. This character set will need to be cleaned afterwards')    
        characters = generate_character_set_from_transcripts(lang)
    
    file_path =os.path.join(os.getcwd(),f'{lang}_characters.txt') 
    with open(file_path,'w+',encoding='utf8') as file_:
        for char in characters:
            file_.write(char)
            file_.write('\n')
    print(f'Characters for {lang} have been saved in {file_path} and will be used for current training. \n Please edit this file and add more characters in order to have a more robust model.')
   
    return file_path 


