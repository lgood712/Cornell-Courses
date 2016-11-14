from __future__ import division, print_function
import random
import re

import numpy as np
from pandas import DataFrame

from stats import SparseMatrix
from utils import find_sublist
from libner import CRFNER

DEBUG = lambda *args: None
#DEBUG = print

class HiddenMarkovNER(object):
    """Guesses the indices of named entities in test data.

    Args:
        train_data (list of triplets)
        test_data (list of triplets)
        normalize (boolean): normalize rows in matrices
        smoothing (float): k for add-k smoothing
        pos_e (boolean): True if pos_matrix should be used
        
    """
    def __init__(self, train_data=[], test_data=[], normalize=True, smoothing=None, pos_e=False, use_known=False):
        super(HiddenMarkovNER, self).__init__()
        self.t_matrix = None #(transition matrix): rows=NE tags, columns = NE tags, intersections = P(ei|ei-1)
        self.w_matrix = None ##(word emission probabilities): rows=NE tags, columns = words, intersections = P(wi|ei)
        self.pos_matrix = None #(POS emission probabilities): rows=NE tags, columns = POS tags, intersections = P(ti|posi)
        self.train_data = train_data
        self.test_data = test_data
        self.normalize = normalize
        self.smoothing = smoothing
        self.pos_e = pos_e

        
    def populate_counts(self):
        """Updates counts in transition and emission probability matrices.
        Args:
            data [(sentence, POS tags, NER tags)]: a list of (context, sense) pairs
        """
        t_matrix = SparseMatrix()
        w_matrix = SparseMatrix(smoothing=self.smoothing)
        if self.pos_e:
            print('POS included')
            pos_matrix = SparseMatrix(smoothing=self.smoothing)
            for sent, pos_tags, ne_tags in self.train_data:
                t_matrix.update(ne_tags)
                w_matrix.update(ne_tags, sent)
                pos_matrix.update(ne_tags, pos_tags)
            self.pos_matrix = pos_matrix
        else:
            for sent, pos_tags, ne_tags in self.train_data:
                t_matrix.update(ne_tags)
                w_matrix.update(ne_tags, sent)
        self.t_matrix = t_matrix
        self.w_matrix = w_matrix
        return self
    
    def get_test_entities(self):
        """ Returns dictionary of test entity indices by type """
        print('\nTEST')
        test_entities = {'PER':[], 'LOC':[], 'ORG':[], 'MISC':[]}
        #for triplet in test list
        for sent, pos_tags, indices in self.test_data:
            #returns viterbi max likelihood sequence of ne_tags
            ne_seq = self.viterbi(sent,pos_tags,indices)
            #record indices of predicted NE tags in test entities
            begin, e_type, inside = None, None, False
            for i in range(len(ne_seq)):
                if ne_seq[i].startswith('B-'):
                    begin = i  # CHANGED FROM indices[i]
                    e_type = ne_seq[i][2:]
                    inside = True
                if inside and (i==len(ne_seq)-1 or not ne_seq[i+1].startswith('I')):
                    tag = {'string': ' '.join(sent[begin:i+1]),
                           'indices': (indices[begin] ,indices[i])}
                    test_entities[e_type].append(tag)
                    #import IPython; IPython.embed(); import time; time.sleep(.5)
                    begin, e_type, inside = None, None, False
                    
        self.test_entities = test_entities
    
    def viterbi(self, sent, pos_tags, indices):
        """ Returns a list of the most likely sequence of NE tags for a given sentence. 
        Args - sent: sentence, obs: pos_tags, indices: indices """
        vit = [{}] #stores probability of each state at each step
        path = {} #stores most likely path at each step
        states = self.w_matrix.keys() #different states are the different NE tags
        
        #initialize (t == 0)
        if self.pos_e:
            for ne_tag in states:
                vit[0][ne_tag] = (self.t_matrix.distribution('initial').surprisal(ne_tag) +
                                  self.w_matrix.distribution(ne_tag).surprisal(sent[0]) + 
                                  self.pos_matrix.distribution(ne_tag).surprisal(pos_tags[0]))
                path[ne_tag] = [ne_tag]
        else:
            for ne_tag in states:
                vit[0][ne_tag] = (self.t_matrix.distribution('initial').surprisal(ne_tag) +
                                self.w_matrix.distribution(ne_tag).surprisal(sent[0]))
                path[ne_tag] = [ne_tag]

        #viterbi calculations for t > 0
        for t in range(1, len(pos_tags)):
            vit.append({})
            newpath = {}

            for y1 in states:
                pos_emission = self.pos_matrix.distribution(y1).surprisal(pos_tags[t]) if self.pos_e else 0.0
                (surprisal, state) = min((vit[t-1][y0] +
                                          self.t_matrix.distribution(y0).surprisal(y1) +
                                          self.w_matrix.distribution(y1).surprisal(sent[t]) +
                                          pos_emission, y0)
                                         for y0 in states)
                vit[t][y1] = surprisal
                newpath[y1] = path[state] + [y1]
            #replace old path with new
            path = newpath

        #return max likelihood sequence
        n = len(sent) - 1
        #self.print_dptable(vit)
        (surprisal, state) = min((vit[n][y], y) for y in states)
        return path[state]
    
    def print_dptable(self, V):
        """ prints table of viterbi steps """
        s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
        for y in V[0]:
            s += "%.5s: " % y
            s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
            s += "\n"
        print(s)


class BaselineNER(object):
    """ Identifies named entities only if they were listed in the training data. 
     test_entities: dictionary of form {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []} where the lists
    are populated with index ranges of named entities of that type. """
    def __init__(self, train_data, test_data, validation=False):
        super(BaselineNER, self).__init__()
        self.validation = validation
        self.train_data = train_data
        self.test_data = test_data
        self.known_entities = self.get_known_entities()
        #DEBUG(self.known_entities)
        self.test_entities = self.get_test_entities()

    def get_known_entities(self):
        """ Returns dictionary of known entities by type """
        print('TRAIN')
        known_entities = {'PER': set(), 'LOC': set(), 'ORG': set(), 'MISC': set()}
        #for triplet in training list
        for sent, pos_tags, ne_tags in self.train_data:
            entity, e_type = [], None
            for i in xrange(0, len(ne_tags)):
                #end of named entity, append
                if entity and (ne_tags[i] == 'O' or ne_tags[i].startswith('B-')):
                    known_entities[e_type].add(tuple(entity))  # tuple because set
                    #DEBUG('-> entity added = {}'.format(entity))
                    entity = []
                #still inside named entity
                elif entity:
                    entity.append(sent[i])
                    #DEBUG('-> still inside entity = {}'.format(entity))
                #named entity begins
                if ne_tags[i].startswith('B-'):
                    e_type = ne_tags[i][2:]
                    entity = [sent[i]]
                    #DEBUG('-> begin entity = {}'.format(entity))
            # add any leftover named entity
            if entity:
                known_entities[e_type].add(tuple(entity))
                #DEBUG('-> entity added = {}'.format(entity))
        return known_entities
    
    def get_test_entities(self):
        """ Returns dictionary of test entity indices by type """
        print('\nTEST')
        test_entities = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
        #for triplet in test list
        for sent, pos_tags, indices in self.test_data:
            for key in self.known_entities:
                for ne in self.known_entities[key]:
                    matches = find_sublist(ne, sent)
                    #DEBUG('-> matches = {}'.format(matches))
                    for begin, end in matches:
                        tag = {'string': ' '.join(sent[begin:end+1]),
                               'indices': (indices[begin] ,indices[end])}
                        test_entities[key].append(tag)
        return test_entities


def get_data(file_name, validation=0.0):
    """ Returns list of triplets of form (line_1, line_2, line_3) for every 3 lines in a given .txt file 
    (For training data, line_1=sentence, line_2=POS tags, line_3=named entity labels. For test, line_3=word indices) """
    
    with open(file_name) as f:
        #content = f.readlines()
        split_lines = [re.split(r'[ \t]', line.strip()) for line in f]

    # break each example up into a three-tuple of lists
    data = [(split_lines[i], split_lines[i+1], split_lines[i+2])
            for i in range(0, len(split_lines), 3)]

    if validation>0.0:
        valid_set = []
        for x in range(0, int(validation * len(data))):
            ind = random.randint(0, len(data)-1)
            valid_set.append(data.pop(ind))
        return {'data':data, 'valid_set':valid_set}
    return data


def write_results(results, file='test-output.csv'):
    import csv
    with open(file, 'wb+') as out_file:
        fieldnames = ['Type','Prediction']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for e_type, tags in results.iteritems():
            prediction = ' '.join('-'.join(tag['indices']) for tag in tags)
            writer.writerow({'Type':e_type, 'Prediction': prediction})


def get_correct_tags(data):
    """Gets """
    result = {'PER': [], 'LOC': [], 'ORG': [], 'MISC': []}
    #for triplet in training list
    for sent, pos_tags, ne_tags in data:
        e_type = None
        for i, tag in enumerate(ne_tags):
            #end of named entity, append
            if e_type and (tag == 'O' or tag.startswith('B-')):
                ctag = {'string': ' '.join(sent[begin:i]),
                        'indices': (ne_tags[begin], ne_tags[i-1])}
                result[e_type].append(ctag)
                e_type = None
            #named entity begins
            elif tag.startswith('B-'):
                e_type = tag[2:]
                begin = i
        # leftover
        if e_type:
            ctag = {'string': ' '.join(sent[begin:i]),
                    'indices': (ne_tags[begin], ne_tags[i-1])}
            result[e_type].append(ctag)
    return result


def evaluate(our_tags, correct_tags):
    # Because we use sets here, we are testing on unique named entities,
    # thus if London is tagged twice in the validation set as an ORG, but
    # we only catch it once, there will be no loss in recall.
    # This is not ideal of course, but fixing it would require adding
    # indices to the training data.
    our_tags = set(str(x) for x in our_tags)
    correct_tags = set(str(x) for x in correct_tags)
    precision = len(correct_tags & our_tags) / (len(our_tags) if our_tags else 1)
    recall = len(correct_tags & our_tags) / len(correct_tags)
    F1 = 2 * precision * recall / (precision + recall)

    misses = correct_tags - our_tags
    alarms = our_tags - correct_tags

    return np.array([F1, precision, recall]), misses, alarms


def validation(percent, ntrials):
    f_scores = []
    for _ in range(ntrials):
        data = get_data('train.txt', percent)
        training_data, valid_set = data['data'], data['valid_set']

        model_predictions = {}

        baseline = BaselineNER(train_data=training_data, test_data=valid_set)
        model_predictions['baseline'] = baseline.test_entities

        hmm = HiddenMarkovNER(train_data=training_data,
                                test_data=valid_set,
                                smoothing="good_turing",
                                pos_e=True).populate_counts()
        hmm.get_test_entities()
        model_predictions['HMM'] = hmm.test_entities
        

        crf = CRFNER().fit(training_data)
        model_predictions['crf'] = crf.predict(valid_set)
        crf.introspect()
        
        correct_tags = get_correct_tags(valid_set)
        from collections import OrderedDict

        for model, predictions in model_predictions.items():
            result = OrderedDict()
            for ne_type, correct in correct_tags.items():
                our_tags = predictions[ne_type]
                score, misses, alarms = evaluate(our_tags, correct)
                result[ne_type] = score
                f_scores.append([model, ne_type, score[0]])
                #print('\n\n====================================================================')
                #print(ne_type)
                #print('\nMISSES')
                #print(misses)
                #print('\nALARMS')
                #print(alarms)
            print('\n'+model)
            result['avg'] = sum(result.values()) / len(result.values())
            print('type F1    PREC  REC')
            for ne_type, scores in result.items():
                print('{ne_type:4} {scores[0]:5.3f} {scores[1]:5.3f} {scores[2]:5.3f}'
                      .format(**locals()))

    df = DataFrame(f_scores, columns=['Model', 'Entity', 'F1'])
    df.to_pickle('f_scores.pkl')
    return df


def run_tests(baseline=False):
    training_data = get_data('tiny-train.txt')
    test_data = get_data('tiny-test.txt')
    if baseline:
        model = BaselineNER(train_data=training_data, test_data=test_data)
    else:
        model = HiddenMarkovNER(train_data=training_data,
                                test_data=test_data,
                                smoothing="good_turing").populate_counts()
        model.get_test_entities()
    write_results(model.test_entities, 'tiny-output.csv')


def main(clf='HMM'):
    training_data = get_data('train.txt')
    test_data = get_data('test.txt')
    if clf == 'HMM':
        model = HiddenMarkovNER(train_data=training_data,
                                test_data=test_data,
                                smoothing="good_turing",
                                pos_e=True).populate_counts()
        model.get_test_entities()
        write_results(model.test_entities)
    elif clf == 'CRF':
        from libner import LibNER
        model = LibNER().fit(training_data)
        write_results(model.predict(test_data))


if __name__ == '__main__':
    #run_tests(0.15)
    validation(0.15, 10)
    #main('CRF')
    
