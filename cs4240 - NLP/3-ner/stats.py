from __future__ import division
from collections import Counter, defaultdict
import math
#import numpy as np

from utils import cached_property

class SparseMatrix(object):
    """A two dimensional sparse matrix with default 0 values.

    Index it like a numpy array: matrix['foo', 'bar'] is the column labeled
    'bar' of the row labeled 'foo'. All computation intensive properties are
    cached until the next time the matrix is modified.
    """
    def __init__(self, data=None, smoothing=None):
        super(SparseMatrix, self).__init__()
        self.smoothing = smoothing
        self._dict = defaultdict(Counter)
        if data:
            for row, cols in data.items():
                self._dict[row] = Counter(cols)
        self.num_columns = 0
        self.row_counts = Counter()

    @cached_property()
    def normalized_probabilities(self):
        """A dense DataFrame which"""
        #HEY FRED, i didnt use normalized probabilities because my shitty ass computer has issues importing both numpy and pandas
        from pandas import DataFrame
        data = defaultdict(Counter)
        for sense in self:
            for feature in self.column_counts:
                data[sense][feature] = self.distribution(sense).probability(feature)
        print(DataFrame(data).apply(lambda column: column / sum(column)))#np.sum(column))

    def update(self, rows, cols=[]):
        """Updates the counts of `rows` at specified `cols`, and adds to row_counts"""
        #transition matrix
        if len(cols)==0:
            self._dict["initial"][rows[0]] += 1 #counts of ner tags starting sentence
            for i in range(len(rows)-1):
                self._dict[rows[i]][rows[i+1]] +=1
                self.row_counts[rows[i]] += 1
        #emission matrix
        else:
            for i in range(len(rows)):
                self._dict[rows[i]][cols[i]] +=1
                self.row_counts[rows[i]] += 1
        self._delete_cache()

    @cached_property()
    def column_counts(self):
        columns = Counter()
        for row, cols in self._dict.items():
            for col in cols:
                columns[col] += self._dict[row][col]
        return columns

    @cached_property()
    def count_counts(self):
        """Value counts for each row in the matrix.

        value_counts['foo'][c] is the number of elements in the 'foo' row
        that are c. For example, if using the CounterMatrix to represent
        a co-occurence matrix for a bigram model, it would be the number of
        bigrams beginning with 'foo' that occurred two times."""
        count_counts = defaultdict(Counter)
        for item, followers in self._dict.items():
            for f, count in followers.items():
                count_counts[item][count] += 1
            count_counts[item][0] = len(self.column_counts) - sum(count_counts[item].values())
        return count_counts

    @cached_property()
    def good_turing_mapping(self, threshold=3):
        """A dictionary mapping counts to good_turing smoothed counts."""
        total_count_counts = sum(self.count_counts.values(), Counter())
        # total_count_counts[2] is number of bigrams that occurred twice

        def good_turing(c): 
            return (c+1) * (total_count_counts[c+1]) / total_count_counts.get(c, 1)
        gtm = {c: good_turing(c) for c in range(threshold)}
        assert gtm
        return {k: v for k, v in gtm.items() if v > 0}  # can't have 0 counts

    @cached_property()
    def row_distribution(self):
        """A distribution over rows weighted by row.val

        It is weighted by the total of all column values in each row.
        """
        if self.row_counts:
            return Distribution(self.row_counts)
        else:
            counts = {item: sum(follower.values()) 
                      for item, follower in self._dict.items()}
            return Distribution(counts)

    def distribution(self, row):
        """Returns a distribution of column labels for a given row."""
        #if row not in self._dict:
        #    raise ValueError('That row is not in the matrix.')
        try:
            return self._cache['distributions'][row]

        except(KeyError, AttributeError):
            # actual function here
            denominator = self.row_counts[row] if self.row_counts else None
            if self.smoothing:
                if self.smoothing == 'good_turing':
                    smoothing_func = lambda c: self.good_turing_mapping.get(c, c)
                elif type(self.smoothing) in (int, float):
                    smoothing_func = lambda c: c + self.smoothing
                d =  Distribution(self._dict[row], smoothing_func,
                                  self.count_counts[row], total=denominator)
            else:
                d = Distribution(self._dict[row], count_counts=self.count_counts[row], total=denominator)

            # caching boilerplate
            try:
                self._cache['distributions']
            except AttributeError:
                self._cache = {'distributions':{}}
            except KeyError:
                self._cache['distributions'] = {}
            self._cache['distributions'][row] = d
            
            return d

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def keys(self):
        return self._dict.keys()

    def __getitem__(self, key):
        # i.e. matrix[foo, bar]
        if isinstance(key, tuple):
            return self._dict[key[0]][key[1]]
        else:
            return self._dict[key]

    def __setitem__(self, key, value):
        try:
            self._dict[key[0]][key[1]] = value
        except TypeError:
            raise TypeError('You cannot set a row of a CounterMatrix')
        self._delete_cache()

    def __delitem__(self, key):
        del self._dict[key]
        self._delete_cache

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def _delete_cache(self):
        try:
            del self._cache
        except AttributeError:
            pass


class Distribution(object):
    """A statistical distribution based on a dictionary of counts."""
    def __init__(self, counter, smoothing_func=None, count_counts=None, total=None):
        assert counter
        self.counter = counter
        self.smoothing_func = smoothing_func
        if total:
            self.total = total
        elif smoothing_func:
            if not count_counts:
                raise ValueError('Must supply count_counts argument to use smoothing.')
            self.total = sum(smoothing_func(count) * N_count 
                                    for count, N_count in count_counts.items())
        else:
            self.total = sum(counter.values())

    def probability(self, item):
        """The probability of an item being sampled."""
        count = self.counter.get(item, 0)
        if self.smoothing_func:
            smooth_count = self.smoothing_func(count)
            assert smooth_count > 0
            return smooth_count / (self.total)
        else:
            return count / (self.total)
    
    def surprisal(self, item):
        """The negative log probability of an item being sampled.
            Accounts for 0.0 probabilities by adding 0.000...0001 to probabilities. """
        try:
            return - math.log(self.probability(item))
        except:
            #HEY FRED, this is how i adjusted for those unsmoothed zero counts (as well as for transition counts, which actually don't get smoothed)
            #NOTE: this actually should do the job anyway since it will make things super unlikely
            return - math.log(self.probability(item)+1e-67)


if __name__ == '__main__':
    a,b,c = 'abc'
    m = SparseMatrix({a:{b:1, c:2}})
    m[a,b] += 1
    m[a,c] += 2
    print(m[a,b])
    print(m[a])
    print(m.count_counts)
    print(m.distribution(a).probability(c))
    m[a,b] = 4
    print(m.count_counts)
    print(m.distribution(a).probability(c))
    
    
    
    
    
    
    
    
#HEY FRED, penis.

