import pycrfsuite
from utils import Timer, cached_property
import re

SPACY = None

class CRFNER(object):
    """NER using a CRF.

    Args:
        features (str): specifies a function used for extracting features. """
    def __init__(self, features='spacy'):
        self.features = features

    @cached_property()
    def sent2features(self):
        """Returns a function used to extract features from a sentence"""
        if self.features == 'spacy':
            global SPACY
            if not SPACY:
                print('loading spacy')
                from spacy.en import English
                SPACY = English(load_vectors=False)
                print('loaded SPACY')
            return spacy_sent2features


    def fit(self, data):
        """Trains the model on the given data, replacing any existing data."""
        sent2features = self.sent2features  # don't include import time in Timer
        with Timer('CRFNER train time'):
            X = [sent2features(sent) for sent,_,_ in data]
            y = [tags for _,_,tags in data]
            trainer = pycrfsuite.Trainer(verbose=False)
            skipped = 0
            for xseq, yseq in zip(X, y):
                try:
                    trainer.append(xseq, yseq)
                except ValueError as e:
                    skipped += 1
            print('skipped {skipped} items'.format(**locals()))
            trainer.set_params({'feature.possible_transitions': True})
            trainer.train('lib.crfsuite')

            self.tagger = pycrfsuite.Tagger()
            self.tagger.open('lib.crfsuite')
        return self

    def predict(self, data):
        """Returns a dictionary of entities in the test data."""
        test_entities = {'PER':[], 'LOC':[], 'ORG':[], 'MISC':[]}
        for sent, pos_tags, indices in data:
            try:
                ne_seq = self.tagger.tag(self.sent2features(sent))
            except Exception as e:
                print(e)
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
                    begin, e_type, inside = None, None, False

        return test_entities

    def introspect(self):
        """Prints a summary of the model's knowledg.

        Taken from:
        http://nbviewer.ipython.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
        """
        from collections import Counter
        info = self.tagger.info()

        def print_transitions(trans_features):
            for (label_from, label_to), weight in trans_features:
                print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

        print("Top likely transitions:")
        print_transitions(Counter(info.transitions).most_common(15))

        print("\nTop unlikely transitions:")
        print_transitions(Counter(info.transitions).most_common()[-15:])

        def print_state_features(state_features):
            for (attr, label), weight in state_features:
                print("%0.6f %-6s %s" % (weight, label, attr))    

        print("\nTop positive:")
        print_state_features(Counter(info.state_features).most_common(20))

        print("\nTop negative:")
        print_state_features(Counter(info.state_features).most_common()[-20:][::-1])


def spacy_sent2features(sent):
    """Returns a list of feature sets, one for each token in sent."""
    length = len(sent)
    uni_sent = clean(unicode(' '.join(sent)))
    doc = SPACY(uni_sent)
    if len(doc) != length:
        print('\n')
        print(' '.join(sent))
        print(' '.join(t.string for t in doc))
    return [word2features(doc, i) for i in range(len(doc))]



def clean(txt):
    """Scrub the input so that spacy produces the same tokens as the files."""
    txt = re.sub(r'\d+', 'NUMBER', txt)
    txt = re.sub(r'[^a-zA-Z0-9 ]', 'PUNCT', txt)
    txt = txt.replace('cannot', 'can')
    return txt


def word2features(doc, head_idx):
    """Returns a set of features for the word at given index in doc."""
    
    def collocation_features(feature):
        features = set()
        for shift in range(-2, 3):
            try:
                idx = head_idx + shift
                if idx < 0: 
                    continue  # don't loop to end of sentence
                val = getattr(doc[idx], feature)
                # unique feature id: e.g. 'col_-1_cluster_1230'
                features.add('_col_{shift}_{feature}_{val}_'.format(**locals()))
            except IndexError:
                pass
        return features

    def dependency_features(feature):
        features = set()
        target = doc[head_idx]
        features.add('_dep_target_role_{}_'.format(target.dep_))  # e.g. DOBJ
        head_val = getattr(target.head, feature)  # e.g. lemma of head
        features.add('_dep_head_{feature}_{head_val}_'.format(**locals()))
        for child in target.children:
            role = child.dep_
            val = getattr(child, feature)
            features.add('_dep_{role}_{feature}_{val}_'.format(**locals()))
        return features

    def orthographic_features():
        features = {}
        word = doc[head_idx].string
        features['upper'] = word.isupper()
        features['title'] = word.istitle()
        return set('_{key}_{val}_'.format(**locals()) for key, val in features.items())

    # could add or remove lines below
    feature_ids = ({'bias'}
                   | collocation_features('cluster')
                   | collocation_features('lower_')
                   | collocation_features('tag_')  # part of speech
                   #| collocation_features('is_oov')  # out of vocabulary
                   | dependency_features('cluster')
                   | orthographic_features()
                   )
    return feature_ids