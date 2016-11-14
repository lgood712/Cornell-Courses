import xml.etree.ElementTree as etree
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(font_scale=1.3)

from stats import SparseMatrix

# for debugging, we use cached context_features because SPACY
# takes a long time to import (it loads a ton of data for parsing)
SPACY = True 
if SPACY:
    from spacy.en import English
    SPACY = English(entity=False, load_vectors=False)
else:
    import pickle
    with open('training_context_features.pkl', 'rb') as f:
        FEATURES = pickle.load(f)



DEBUG = lambda *args: None
#DEBUG = print


class Disambiguator(object):
    """Guesses the sense of a word in context.

    Args:
        smoothing (float): k for add-k smoothing
        features (iterable): features to use, defaults to all
        exclude (iterable): features to not use
        drop_threshold (int): minimum occurrences for a feature to be used
        baseline (bool): if True, use only baserate to classify
    """
    def __init__(self, normalize=True, smoothing=0.18, drop_threshold=0,
                 features=[], exclude=[], baseline=False):
        super(Disambiguator, self).__init__()
        self.matrix = None
        self.normalize = normalize
        self.smoothing = smoothing
        self.drop_threshold = drop_threshold
        self.features = features
        self.exclude = exclude
        self.baseline = baseline

    def classify(self, context):
        """Returns the most likely sense of the head word in a context."""
        if self.matrix is None:
            raise ValueError('Disambiguator must be trained before it can classify.')
        features = extract_features(context, self.features, self.exclude)
        features &= set(self.matrix.column_counts)  # ignore unknown features
        sense_surprisals = {sense: self.matrix.row_distribution.surprisal(sense)
                            for sense in self.matrix}  # priors = baserates
        # add surprisal for every feature to every sense's surprisal
        if not self.baseline:
            for sense in self.matrix:
                for feature in features:
                    if self.normalize:
                        surprisal = - np.log(self.matrix.normalized_probabilities[sense][feature])
                    else:
                        surprisal = self.matrix.distribution(sense).surprisal(feature)
                    sense_surprisals[sense] += surprisal
        return min(sense_surprisals, key=sense_surprisals.get)
        
    def fit(self, data):
        """Updates the sense-feature probability matrix.

        Args:
            data ((str, str) list): a list of (context, sense) pairs
        """
        # matrix has senses as rows, features as columns, and counts as elements
        matrix = SparseMatrix(smoothing=self.smoothing)
        for context, sense in data:
            features = extract_features(context, self.features, self.exclude)
            matrix.update_row(sense, features)
        if self.drop_threshold:
            matrix.drop_weak_columns(self.drop_threshold)  # ignore rare features

        self.matrix = matrix
        return self


def extract_features(context, features=[], exclude=[]):
    """Returns a set of feature IDs for this context.

    Args:
        context: a string with the target word wrapped with <head>
        features: a list of feature_ids or parts of feature_ids. Any
            feature that contains one of the strings will be included.
        exclude: overrides features, matches will not be included.
    """

    if isinstance(features, str):
        features = [features] 
    features = ['_{}_'.format(f) for f in features]
    if isinstance(exclude, str):
        exclude = [exclude] 
    exclude = ['_{}_'.format(f) for f in exclude]

    if not SPACY:
        # we use cached features for debugging
        feature_ids = FEATURES[context]
    else:
        # find the head token
        tokens = SPACY.tokenizer(context)
        for i, token in enumerate(tokens):
            if 'head>' in token.text:
                head_idx = i 
                break
        assert head_idx

        # remove tags
        head = tokens[head_idx].text
        start, stop = head.index('>') + 1, head.rindex('</')
        head = head[start:stop]
        clean_context = (tokens[:head_idx-1].text
                         + ' ' + head + ' '
                         + tokens[head_idx+2:].text)
        doc = SPACY(clean_context)
        head_idx -= 1  # because we remove the previous '<' token

        # each of these functions extracts a feature (e.g. 'lemma') from
        # the words in the specified locations (e.g. dependencies)
        def cooccurrence_features(feature):
            features = set()
            for token in doc:
                val = getattr(token, feature)
                features.add('_coo_{feature}_{val}_'.format_map(locals()))
            return features

        def collocation_features(feature):
            features = set()
            for i in range(-2, 3):
                val = getattr(doc[head_idx + i], feature)
                # unique feature id: e.g. 'col_-1_cluster_1230'
                features.add('_col_{i}_{feature}_{val}_'.format_map(locals()))
            return features

        def dependency_features(feature):
            features = set()
            target = doc[head_idx]
            features.add('_dep_target_role_{}_'.format(target.dep_))  # e.g. DOBJ
            head_val = getattr(target.head, feature)  # e.g. lemma of head
            features.add('_dep_head_{feature}_{head_val}_'.format_map(locals()))
            for child in target.children:
                role = child.dep_
                val = getattr(child, feature)
                features.add('_dep_{role}_{feature}_{val}_'.format_map(locals()))
            return features

        # could add or remove lines below
        feature_ids = (set()
                       | cooccurrence_features('lemma_')
                       | cooccurrence_features('cluster')
                       | collocation_features('lemma_')
                       | collocation_features('cluster')
                       | collocation_features('tag_')  # part of speech
                       | dependency_features('lemma_')
                       | dependency_features('cluster')
                       )

    if features:
        # only keep those listed
        feature_ids = set(f for f in feature_ids
                          if  any(keep in f for keep in features))
    if exclude:
        # exclude those listed
        feature_ids = set(f for f in feature_ids
                          if not any(excl in f for excl in exclude))
    
    return feature_ids


def get_data(xml_file, train=True, validation=0.0, ignore=[]):
    """Returns a dictioary with training and validation data."""
    with open(xml_file,'r') as f:
        xml_as_string = '<xml>\n' + f.read() + '\n</xml>'

    #replace 'not well-formed' XML occurrences of '&'
    xml_as_string = xml_as_string.replace('&', '&amp;')

    #retrieve element tree
    root = etree.fromstring(xml_as_string)
    
    def cleaned_content(tag):
        """Given element tree element, returns inner text with 
        XML tags included and line breaks ('\n') removed."""
        return (tag.text + ''.join(str(etree.tostring(e)) for e in tag)).replace('\n', '')

    #{target: [(contex, sense),...]} tuple dictionary
    data = {}
    #iterate through target words
    for target in root.iter(tag='lexelt'):
        data[target.attrib['item']] = []
        #iterate through instance elements of target
        for elem in target.iter(tag='instance'):
            #retrieve context of instance
            context = cleaned_content(elem.find('context'))

            if train:
                #retrieve list of senseids
                senses = [answer.attrib['senseid'] for answer in elem.iter(tag='answer')
                          if answer.attrib['senseid'] not in ignore]
                if senses:
                    sense = ' '.join(senses)
                    data[target.attrib['item']].append((context, sense))
            else:
                data[target.attrib['item']].append((context, elem.attrib['id']))
    if train:
        #validation set
        valid_set = {}
        for target in data.keys():
            target_data = data[target]
            valid_set[target] = []
            for x in range(0, int(validation * len(target_data))):
                ind = np.random.randint(0, len(target_data)-1)
                valid_set[target].append(target_data.pop(ind))
        return {"training":data, "validation":valid_set}
    else:
        return data


def classify_test():
    data = get_data('training-data.data')
    training_data, validation_set = data['training'], data['validation']
    models = {target: Disambiguator().fit(training_data[target])
              for target in training_data.keys()}
    with open('test-output21.csv', 'w+') as out_file:
        out_file.write('Id,Prediction\n')
        test_data = get_data('test-data.data', train=False)
        for target, contexts in test_data.items():
            for context, instance_id in contexts:
                classification = models[target].classify(context)
                out_file.write('{instance_id},{classification}\n'.format_map(locals()))


def validate(models, validation_set):
    """Returns performance of a set of models on a validation set."""
    denom = 0
    correct = 0
    for target in validation_set:
        for context, sense in validation_set[target]:
            denom += 1
            classification = models[target].classify(context)
            if classification == sense:
                correct += 1
            else:
                DEBUG('misclassified {sense} as {classification}'.format_map(locals()))
    #print('VALIDATION CORRECT:', correct / denom)
    return(correct / denom)


def test_parameters(ntrials, param, values, **kwargs):
    """Performs cross validation with random subsets of the data."""
    pkl_file = '{param}_{kwargs}_alt.pkl'.format_map(locals())
    performance = {v: [] for v in values}
    # ideally we would use k-fold cross validation, but for simplicity
    # we simly use random validation sets
    for _ in range(ntrials):
        data = get_data('training-data.data', validation=0.1, ignore=[])
        training_data, validation_set = data['training'], data['validation']
        for val in performance:
            kwargs[param] = val
            models = {target: Disambiguator(**kwargs).fit(training_data[target])
                      for target in training_data.keys()}
            
            correct = validate(models, validation_set)
            performance[val].append(correct)

    df = pd.DataFrame(performance)
    print(df)
    df.to_pickle(pkl_file)
    return df

def exclude_plot(df, features):
    df = df.copy()
    full = df.pop('None ')
    mdf = pd.melt(df, var_name='excluded', value_name='accuracy')
    ax = sns.barplot('excluded', 'accuracy', data=mdf, order=features[1:],
                     color=sns.color_palette()[0])
    ax.set_xticklabels(['lemma', 'cluster', 'tag', 'co-occurrence',
                       'co-location', 'dependency'], rotation=20)
    ax.set_ylim(0.4,0.7)
    sns.plt.axhline(full.mean(), color=sns.color_palette()[1],
                    linestyle='--', label='full model')
    sns.plt.show()


def boxplot(df, var_name, **kwargs):
    mdf = pd.melt(df, var_name=var_name, value_name='accuracy')
    ax = sns.boxplot(var_name, 'accuracy', data=mdf, **kwargs)
    sns.plt.show()

def pointplot(df, var_name, labels=None):
    mdf = pd.melt(df, var_name=var_name, value_name='accuracy')
    ax = sns.pointplot(var_name, 'accuracy', data=mdf, ci=95, markers='')
    if labels:
        ax.set_xticklabels(labels)
    sns.plt.show()


def save_features(data, file):
    context_features = {}
    for target, examples in data.items():
        for context, sense in examples:
            context_features[context] = extract_features(context)
    with open(file, 'wb+') as f:
        import pickle
        pickle.dump(context_features, f, protocol=2)


def make_plots(validation_rounds=5):
    smoothing_values, labels = zip(*[(2**-e, r'$2^{-%s}$' % e) 
                                     for e in range(10, 0, -1)])

    smoothing = test_parameters(validation_rounds, 'smoothing', 
                                smoothing_values, normalize=False)

    norm_smoothing = test_parameters(validation_rounds, 'smoothing',
                                     smoothing_values, normalize=True)


    baseline = test_parameters(validation_rounds, 'baseline', [True, False])
    
    features = ['None ', 'lemma', 'cluster', 'tag','coo', 'col', 'dep']
    exclude = test_parameters(validation_rounds, 'exclude', features)

    boxplot(baseline, 'baseline')
    pointplot(smoothing, 'smoothing', labels)
    pointplot(norm_smoothing, 'smoothing', labels)
    exclude_plot(exclude, features)

if __name__ == '__main__':
    make_plots(10)
