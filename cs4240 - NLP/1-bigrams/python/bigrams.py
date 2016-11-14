""""
Project 1 for CS 4740
Fred Callaway, Kang-Li Chen
"""
import bisect
from collections import Counter, defaultdict
import itertools
import math
import random
from sklearn.svm import SVC
from typing import Dict, List

import time

import preprocess
from utils import cached_property, flatten, split
from functools import lru_cache


class BigramModel(object):
    """A bigram language model.

    Args:
        tokens: a list of tokens to train on.
        smoothing: should we use Good-Turing smoothing?
        track_rare: 'first' or an int. If 'first' we mark the first occurrence of 
            each word as a rare word. If an int, it serves as a frequency cutoff
            for being marked as a rare word."""
    def __init__(self, tokens: List[str], smoothing=True, track_rare=1):
        assert tokens
        self.tokens = tokens
        self.smoothing = smoothing
        self.track_rare = track_rare

        if track_rare is 'first':
            # first occurcence of a word is replaced with UNKNOWN_TOKEN
            seen_words = {'SENTENCE_BOUNDARY'}  # don't mess with this
            for i in range(len(self.tokens)):
                    word = self.tokens[i]
                    if word not in seen_words:
                        self.tokens[i] = 'UNKNOWN_TOKEN'
                    seen_words.add(word)
        elif type(track_rare) is int:
            # words occurring fewer than track_rare times are replaced
            occurences = Counter(self.tokens)
            for i, token in enumerate(self.tokens):
                if occurences[token] <= self.track_rare:
                    tokens[i] = 'UNKNOWN_TOKEN'

        self._cooccurrence_matrix = CounterMatrix(self.tokens, smooth=self.smoothing)

    def predict_next(self, token: str) -> str:
        """Returns a token from distribution of tokens that follow this token."""
        return self._cooccurrence_matrix.distribution(token).sample()

    def surprisal(self, token: str, follower: str) -> float:
        """Returns the negative log probability of `follower` following `token`

        -log p(follower_i | token_{i-1})"""
        try:    
            dist = self._cooccurrence_matrix.distribution(token)
        except KeyError:
            dist = self._cooccurrence_matrix.distribution('UNKNOWN_TOKEN')
        return dist.surprisal(follower)

    def probability(self, token: str, follower: str) -> float:
        """Returns the probability of `follower` following `token`

        -log p(follower_i | token_{i-1})"""
        return self._cooccurrence_matrix.distribution(token).probability(follower)

    def unigram_probability(self, token: str) -> float:
        return self._cooccurrence_matrix.unigram_distribution.probability(token)

    def generate_sentence(self, initial="") -> str:
        """Returns a randomly generated sentence.

        Optionally, the beginning of the sentence is given."""
        words = initial.split()
        if not words:
            # the first word should occur frequently after a sentence boundary
            words.append('SENTENCE_BOUNDARY')

        for i in range(30):  # keep sentences from getting too long
            next_word = self.predict_next(words[-1])
            # avoid generating sentences with UNKNOWN_TOKEN
            while next_word == 'UNKNOWN_TOKEN':
                # We sample under the condition that no token in the sentence
                # is unknown by resampling when the condition is violated.
                next_word = self.predict_next(words[-1])
            if next_word == 'SENTENCE_BOUNDARY':
                break
            else:
                words.append(next_word)
            if i == 29:
                words.append('...')
        words.pop(0)  # remove SENTENCE_BOUNDARY
        return ' '.join(words) + '\n'

    def perplexity(self, tokens: List[str]) -> float:
        """Average surprisal."""
        first_surprisal = self.surprisal('SENTENCE_BOUNDARY', tokens[0])
        total_surprisal = first_surprisal + sum(self.surprisal(tokens[i], tokens[i+1])
                                                for i in range(len(tokens) - 1))

        return math.exp(total_surprisal / (len(tokens)))


class GenreClassifier(object):
    """Classifies genres.

    Args:
        genre_corpora: dict from genres to directories containing training files.
        train_dirs: directories with training files for each genre.
        num_validate: the number of tokens to use for training the svc.
    """
    def __init__(self, genres, train_dirs, num_validate=100000):
        start = time.time()
        self.num_validate = num_validate
        self.genres = genres
        print('-> genres = {}'.format(self.genres))
        self.corpora = [preprocess.dir_to_tokens(file) for file in train_dirs]
        
        training_corpora = [c[:-self.num_validate] for c in self.corpora]
        self.models = [BigramModel(c) for c in training_corpora]
        print('-> train time = {}'.format((time.time() - start)))

        # model baseline is performance on a corpus with all genres
        validation_corpus = flatten(c[-self.num_validate:] for c in self.corpora)
        self.baselines = [model.perplexity(validation_corpus) for model in self.models]
        print('-> perlexity baselines = {}'.format([round(b) for b in self.baselines]))
        
    def classify(self, filename):
        """The genre which the text in a file most likely belongs to.

        Computed as the genre of the model which returns the lowest perplexity
        relative to its average perplexity"""
        tokens = preprocess.parse_book(filename)
        perplexities = [m.perplexity(tokens) for m in self.models]
        print('-> perplexities = {}'.format([round(p) for p in perplexities]))
        over_baseline = [p - b for p, b in zip(perplexities, self.baselines)]
        print('-> over_baseline = {}'.format([round(o) for o in over_baseline]))
        min_index = over_baseline.index(min(over_baseline))
        return self.genres[min_index]

    @cached_property()
    def svc(self):
        """A scikit learn support vector classifier for genres.

        It doesn't work very well at all."""
        validation_corpora = [c[-self.num_validate:] for c in self.corpora]
        split_vc = flatten(split(corpus, 10) for corpus in validation_corpora)
        perplexities = [[m.perplexity(vc) for m in self.models]
                        for vc in split_vc]
        targets = flatten([i]*10 for i in range(len(self.genres)))
        svc = SVC().fit(perplexities, targets)
        return svc
        
    def svc_classify(self, filename):
        """The genre that a file is most likely to belong to.

        Or maybe just always 'history'."""
        tokens = preprocess.parse_book(filename)
        perplexities = [m.perplexity(tokens) for m in self.models]
        return self.genres[self.svc.predict(perplexities)]


class CounterMatrix(object):
    """A two dimensional sparse matrix of counts with default 0 values."""
    def __init__(self, tokens, smooth=False):
        super(CounterMatrix, self).__init__()
        self.smooth = smooth

        self._dict = defaultdict(Counter)
        for i in range(len(tokens) - 1):
            self._dict[tokens[i]][tokens[i+1]] += 1

    def __len__(self):
        return len(self._dict)

    @cached_property()
    def count_counts(self):
        """Value counts for each row in the matrix.

        value_counts['foo'][c] is the number of elements in the 'foo' row
        that are c. For example, if using the CounterMatrix to represent
        a co-occurence matrix for a bigram model, it would be the number of
        bigrams beginning with 'foo' that occurred two times."""
        count_counts = defaultdict(Counter)
        for token, followers in self._dict.items():
            for f, count in followers.items():
                count_counts[token][count] += 1
            count_counts[token][0] = len(self._dict) - sum(count_counts[token].values())
        return count_counts

    @cached_property()
    def good_turing_mapping(self, threshold=5) -> Dict[int, float]:
        """A dictionary mapping counts to good_turing smoothed counts."""
        total_count_counts = sum(self.count_counts.values(), Counter())
        # total_count_counts[2] is number of bigrams that occurred twice

        def good_turing(c): 
            return (c+1) * (total_count_counts[c+1]) / total_count_counts.get(c, 1)
        gtm = {c: good_turing(c) for c in range(threshold)}
        return {k: v for k, v in gtm.items() if v > 0}  # can't have 0 counts

    @cached_property()
    def unigram_distribution(self):
        """The probability of each token occurring irrespective of context."""
        counts = {token: sum(follower.values()) 
                  for token, follower in self._dict.items()}
        return Distribution(counts)

    @lru_cache(maxsize=100000)  # caches 100,000 most recent results
    def distribution(self, token):
        """Returns next-token probability distribution for the given token.

        distributions('the').sample() gives words likely to occur after 'the'"""
        if token not in self._dict:
            token = 'UNKNOWN_TOKEN'  # yes, yes, bad coupling I know...
        if self.smooth:
            smoothing_dict = self.good_turing_mapping
            return Distribution(self._dict[token], smoothing_dict,
                                self.count_counts[token])
        else:
            if self._dict[token]:
                return Distribution(self._dict[token])
            else:
                # no information -> use unigram
                return self.unigram_distribution


class Distribution(object):
    """A statistical distribution based on a dictionary of counts."""
    def __init__(self, counter, smoothing_dict={}, count_counts=None):
        assert counter
        self.counter = counter
        self.smoothing_dict = smoothing_dict

        # While finding the total, we also track each
        # intermediate total to make sampling faster.
        self._acc_totals = list(itertools.accumulate(counter.values()))
        self.total = self._acc_totals[-1]

        # Smoothing only applies to surprisal, not sampling so we maintain
        # a separate total that accounts for the smoothed counts
        if smoothing_dict:
            if not count_counts:
                raise ValueError('Must supply count_counts argument to use smoothing.')
            self.smooth_total = sum(smoothing_dict.get(count, count) * N_count 
                                    for count, N_count in count_counts.items())
        else:
            self.smooth_total = None

    def sample(self):
        """Returns an element from the distribution.

        Based on ideas from the following article:
        http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python"""
        rand = random.random() * self.total

        # Perform a binary search for index of highest number below rand.
        # index will thus be chosen with probability =
        # (self._acc_totals[i] - self._acc_totals[i-1]) / self.total
        index = bisect.bisect_right(self._acc_totals, rand)
        tokens = list(self.counter.keys())
        return tokens[index]

    def probability(self, item):
        """The probability of an item being sampled."""
        count = self.counter.get(item, 0)
        if self.smoothing_dict:
            smooth_count = self.smoothing_dict.get(count, count)
            assert smooth_count > 0
            return smooth_count / self.smooth_total
        else:
            return count / self.total
    
    def surprisal(self, item):
        """The negative log probability of an item being sampled."""
        return - math.log(self.probability(item))




def test_genre_classifier(models=[]):
    print('\nTesting genre classifier')
    genres = ['children', 'crime', 'history']
    train_dirs = [preprocess.genre_directory(g) for g in genres]
    clf = GenreClassifier(genres, train_dirs)

    for genre in genres:
        directory = preprocess.genre_directory(genre, test=True)
        for file in preprocess.get_text_files(directory):
            print('%s book classified as %s' % (genre, clf.classify(file)))


def print_example_sentences(prompt='I love when', n=3):
    print('\nPrinting example sentences.')
    genres = ['children', 'crime', 'history']
    models = [BigramModel(preprocess.get_corpus(g)) for g in genres]
    for g, m in zip(genres, models):
        print('Example sentences for a model trained on %s books:' % g)
        print(m.generate_sentence(prompt))
        for _ in range(n):
            print(m.generate_sentence())
        print('-----------------------------------------')


def test_unknown_methods(genre=True, perplexity=True):
    genres = ['children', 'crime', 'history']
    data = defaultdict(list)
    for track_rare in ['first', 1, 2, 3, 6, 12]:
        if perplexity:
            models = [BigramModel(preprocess.get_corpus(g), track_rare=track_rare)
                      for g in genres]
            unknown_probabilities = [m.unigram_probability('UNKNOWN_TOKEN')
                                     for m in models]
            perplexities = [m.perplexity(preprocess.get_corpus(g, test=True))
                            for m, g in zip(models, genres)]
            data['track_rare'].append(track_rare)
            data['unknown_probabilities'].append(unknown_probabilities)
            data['perplexities'].append(perplexities)

        if genre:
            test_genre_classifier(models=models)

    if data:
        # plotting
        from pandas import DataFrame
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.style.use('ggplot')

        df = DataFrame(data)
        df['perplexity'] = df['perplexities'].apply(lambda x: sum(x)/len(x))
        df['unknown_probability'] = df['unknown_probabilities'].apply(lambda x: sum(x)/len(x))
        df['unknown_probability'] = df['unknown_probability'] * 10000
        plt.figure(); df.plot(x='track_rare', y=['perplexity', 'unknown_probability'])
        plt.savefig('unknown_tokens.svg')
        return DataFrame(data)






def main():
    print('WARNING, this will take a while.')
    print_example_sentences()
    test_genre_classifier()
    test_unknown_methods(genre=False)


""" 
We have them the signal for a large proportion of stories about in the
city of Clodius , the administrator of peoples , fought with liabilities
through the battle-field .

THEODORIC THE VISIGOTH ( Velleius Paterculus , distinguished for reform , time
; but the south , and effeminate youths of the stories about 200 horse born
free men , few ...

Clovis resolved to the greatest ages of sixty without result would be struck
the Nile , devoured by speaking , where the agrarian law were made himself had
a collective : ...

At this the translators render itself , p.

Enviously, she cried out their fortune ; [ 1091 ] The Carthaginians were the
affairs of the governing and bitter period , _Cæsar_ , and Antemnæ , did a
tender his son Edward . ...

Lurking in the shadows , _Topogr .

She paints Roman colonies -- Festus , which , _Letters to contest if one hand
was led his rival .
"""


if __name__ == '__main__':
    main()
    #test()