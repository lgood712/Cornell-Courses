import codecs
import os

import nltk.data

from utils import flatten


def tokenize(text):

    # insert SENTENCE_BOUNDARY markers at locations chosen by a punkt tokenizer
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sent_detector.tokenize(text)
    marked_text = ' SENTENCE_BOUNDARY '.join(sents)

    tokens = nltk.word_tokenize(marked_text)
    if tokens[-1] is not 'SENTENCE_BOUNDARY':
        tokens.append('SENTENCE_BOUNDARY')

    return tokens

def parse_book(text_file):
    with codecs.open(text_file, 'r', encoding='utf-8', errors='ignore') as myfile:
        text = myfile.read()
    
    return tokenize(strip_gutenberg(text))

def dir_to_tokens(directory):
    test_files=get_text_files(directory)
    if not test_files:
        raise OSError('No text files found in directory: {}'.format(directory))
    tokenized_books = [parse_book(tf) for tf in test_files]
    return flatten(tokenized_books)

def genre_directory(genre, test=False):
    return '../books/%s_books/%s'% ('test' if test else 'train', genre)
    
def get_corpus(genre, test=False):
    return dir_to_tokens(genre_directory(genre, test))

def get_text_files(directory):
    files = []
    for path, _, base_names in os.walk(directory):
        for bn in base_names:
            if bn.endswith('.txt'):
                files.append(path + '/' + bn)
    return files

def strip_gutenberg(text):
    start = text.find('Contents') + len('Contents')
    end = text.find('End of the Project')
    return text[start:end].strip()
