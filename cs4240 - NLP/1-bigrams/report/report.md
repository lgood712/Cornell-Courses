NLP Project 1: Language modeling
================================

#### by Fred Callaway and Luke Goodman

---

Model components
================

Preprocessing
-------------

As input, we take directories containing e-books in plain text format, sourced from Project Gutenberg. We first strip extraneous text inserted by Project Gutenberg. We then pass the text through a pre-trained `nltk` punkt tokenizer to mark sentence boundaries. Finally, we tokenize words using the standard `nltk.word_tokenize()`. The `preprocess` module contains small functions for this process.

Calculating probabilities
-------------------------

At its core, a bigram model is a function from tokens to distributions over tokens. Let $\#(x, y)$ be the number of times $y$ follows $x$ in the training corpus. We can define a distribution $p_x$ for a token $x$ as follows.

$$ p*x(y) = \frac{ \#(x, y) }{ \sum*{t} \#(x, t) } $$

These values are both straightforward to find if we have a co-occurrence matrix. We would simply look up the value at $C*{xy}$ and divide by the sum of $C_x$. Creating the co-occurrence matrix is straightforward as well. We use a `defaultdict` of `Counter`s, which are themselves just `defaultdict`s with default 0 values. This implements a _dictionary of keys sparse matrix.* Each `Counter` corresponds to a row in the co-occurrence matrix.

```python
# from CounterMatrix.__init__
self._dict = defaultdict(Counter)
for i in range(len(tokens) - 1):
    self._dict[tokens[i]][tokens[i+1]] += 1
```

Once we have this co-occurrence matrix, we can create a distribution for each token as follows (simplified from final implementation). Note that the line where we find `self.total` i.e $\sum*{t} \#(x, t)$ requires iterating over the entire row. But that's totally fine because the matrix is sparse. We can assume almost every token has a zero probability of following this one. And we'll always be able to assume that _so there's nothing to worry about, okay?!*

```python
class Distribution(object):
    """A statistical distribution based on a dictionary of counts."""
    def __init__(self, counter):
        self.counter = counter
        self.total = sum(counter.values())  # len(counter) better be low!

    def probability(self, item):
        """The probability of an item being sampled."""
        count = self.counter[item]  # will be 0 if item not in counter
        return count / self.total
```

Random sentence generation
--------------------------

A bigram model captures the probability distribution of a corpus, thus it is capable of sampling from this distribution. Note the use of resampling to avoid getting `"UNKNOWN_TOKEN"` in the output sentences. In the current implementation, we truncate overly long sentences, thus we are really sampling from the distribution of the first ~30 words of all valid sentences.

```python
# taken from BigramModel

def predict_next(self, token):
    """Returns a token from distribution of tokens that follow this token."""
    return self._cooccurrence_matrix.distribution(token).sample()

def generate_sentence(self, initial=""):
    """Returns a randomly generated sentence.

    Optionally, the beginning of the sentence is given."""
    words = initial.split()
    if not words:
        # the first word should occur frequently after a sentence boundary
        words.append(self.predict_next('SENTENCE_BOUNDARY'))
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
    return ' '.join(words) + '\n'
```

### Examples

I used `print_example_sentences()` to generate one sentence with a prompt "I love when" and then 50 more unprompted sentences. I include the prompted sentence and my favorite of the other 50.

**Children**

-	*I love when he raised himself into a new guardian of low whistle , live. in dressed him before they rode against themselves were at him from his horse and the former kings ...*
-	*We are all chances to become as a discourse .*

**Crime**

-	*I love when he had reached you have no use . '' said I.*
-	*You 'll have just now ; but one -- there 's the room , ''*

**History**

-	*I love when he would have the condition only one day , always be sanctioned also is called , when in his expectation of themselves .*
-	*They therefore raised him when his feet were exposed to others .*

And to top it off, from the children: *You go with such nonsense !*

Perplexity
----------

In addition to sampling from the learned distribution, a generative model can assign a probability to any element that the distribution covers: in this case, any sequence of tokens in the corpus. We apply the chain rule, which is greatly simplified by the Markov assumption. Let $q$ be the model's probability distribution. We calculate the probability of a sequence $A$ as follows:

$$ q\Bigg(\bigcap*{k=1}^n A_k\Bigg) = \prod*{k=1}^n q\Bigg(A*k \,\Bigg|\, \bigcap*{j=1}^{k-1} A*j\Bigg) = \prod*{k=1}^n q(A*k | A*{k-1}) $$

To avoid underflow errors (among other theoretical motivations), we use surprisal (i.e. negative log probability) in the place of probability. Because $log(xy) = log(x) + log(y)$, the product becomes a sum. And because we want our metric to be neutral to the length of the sequence, we take the average surprisal. Finally, in order to make differences in models sound more impressive to journal editors, we take the exponentiation of average surprisal to get perplexity.

$$\exp \frac{1}{N} -\sum*{k=1}^N \log q(A_k | A*{k-1})$$

I don't include any code snippets because it's mostly just math in python form.

Smoothing and unknown words
---------------------------

So, everything is looking great. We can generate sentences and even assign probabilities to new sentences! Look, see how--wait, `ValueError: math domain error`--what the heck? It looks we tried to take the logarithm of 0... Our `defaultdict` matrix just did its job and this is how it gets repaid!

For any open vocabulary modeling task, we need a way to handle never-before-seen word transitions. These transitions come in two flavors: bigrams with a new word, or bigrams with old words that never occurred together.

### Unknown words

There are a few ways to handle the first case. We do not just want to assign some low probability to any bigram with a new word. Some words will be commonly followed by new words, while others will almost never (e.g. "the" and "go" respectively).

We can learn about the distribution of new words by treating them like rare words in the test corpus. One way to do this is to replace the first occurrence of every word with `UNKNOWN_TOKEN`. For the model, that word was unknown at the time, so this makes some sense. However, this will result in common words being marked as unknown once, which may distort the distribution. Thus, we also explored a strategy of marking words by their absolute occurrence count. Below is a plot showing perplexity and unigram probability of `UNKNOWN_TOKEN` (scaled) for the "first occurrence" strategy and different cutoff frequency for rare words. This plot is generated by `test_unknown_methods()`

![Effect of track_rare on perplexity and `UNKNOWN_TOKEN` probability](unknown_plot.png)

It is expected that perplexity will inversely correlate with unknown probability because this replaces surprising tokens with one increasingly common token. In the limit, every token is unknown, and the perplexity is 0. However, the fact that perplexity decreases from "first occurrence" to "occurs once" while unknown probability decreases as well indicates that the cutoff method is genuinely better than the "first occurrence" method.

### Unseen bigrams

Handling unseen bigrams is a specific instance of the more general problem of smoothing to avoid overfitting. This ammounts to moving some of the probability mass from seen bigrams to unseen bigrams. We can view this as the model sacrificing some of its knowledge about the training set to achieve better generalization.

We implemented Good-Turing smoothing for all tokens that occur less than five times. We are able to apply the smoothing to only certain tokens wihout invalidating the distribution because we adjust counts, not the probabilities themselves. Below is our implementation of the following function, where $c$ is a bigram count and $N_c$ is the number of bigrams that have that count.

$$GoodTuring(c) = (c+1) \frac{ N_{c+1} }{ N_c }$$

```python
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
        count_counts[token][0] = len(self) - sum(count_counts[token].values())
    return count_counts

def good_turing_mapping(self, threshold=5):
    """A dictionary mapping counts to good_turing smoothed counts."""
    total_count_counts = sum(self.count_counts.values(), Counter())
    # total_count_counts[2] is number of bigrams that occurred twice

    def good_turing(c):
        return (c+1) * (total_count_counts[c+1]) / total_count_counts.get(c, 1)

    gtm = {c: good_turing(c) for c in range(threshold)}
    return {k: v for k, v in gtm.items() if v > 0}  # can't have 0 counts
```

An example Good-Turing mapping is $0 \mapsto 0.00051, 1 \mapsto 0.42, 2 \mapsto 1.32, 3 \mapsto 2.12, 4 \mapsto 3.13$. Now we simply apply this function to every element in our cooccurrence matrix and proceed as before, provided that we have a supercomputer or lots of time. Applying this mapping destroys the sparsity of our matrix, something our performance depends heavily on. Thus, we need a way to apply the Good-Turing mapping as a filter between the raw counts from the matrix and the probability calculation to find:

$$ p*x(y) = \frac{ GoodTuring(\#(x, y)) }{ \sum*{t} GoodTuring(\#(x, t)) } $$

Unfortunately, the denominator takes linear time with the number of unique tokens, which is means quadratic time overall because we may have to calculate this function for every $x$. To avoid this, we cae use the previously computed `count_counts`. We restate $GoodTuring(\#(x, t))$ as $\sum_c GoodTuring(c) N_c$ (proof can't fit in 6 pages). Note that `count_counts` doesn't suffer from the same time complexity problem because it calculates zero counts as the total number of possible bigrams minus the number of seen bigrams.

```python
# from Distribution.__init__()
self.smooth_total = sum(smoothing_dict.get(count, count) * N_count
                        for count, N_count in count_counts.items())
```

Genre classification
====================

The first NLP application we applied our model to was a genre classifying task. The basic idea is very intuitive: train a model on each of the genre training sets and then find the perplexity of each model on a test book. We expect that the models will have learned some domain specific knowledge, and will thus be least *perplexed* by the test book.

However, we found that this approach did not work: the crime model did too well. Our first thought was to blindly throw machine learning at the problem (see `svc_classify()`), but this worked even worse than our simple rule. So we decided to examine the perplexity matrix in the hopes of diagnosing the problem:

|          | children | crime   | history |
|----------|----------|---------|---------|
| children | 580      | **575** | 2515    |
| crime    | 425      | **279** | 1342    |
| crime    | 484      | **376** | 1462    |
| history  | **1039** | 1131    | 1389    |

Table: Perplexities for each model on the test books. These values are from using the "first occurrence" unknown_token strategy

Although classification performance is at chance level, we notice that there is a pattern to the results. Each model seems to have a selective advantage on its own genre. To quantify this, we compute baseline perplexities for each model with a mixed genre validation corpus. We apply our original "lowest perplexity" rule, but using the difference between each genre's perplexity and its baseline. The model now achieves perfect performance. However, we cannot take our results too seriously because we used the test corpora as a development corpora.

\newpage

|               | children | crime    | history |
|---------------|----------|----------|---------|
| mix of genres | 710      | **679**  | 1393    |
| children      | **-130** | -104     | 1122    |
| crime         | -285     | **-400** | -51     |
| crime         | -226     | **-303** | 69      |
| history       | 330      | 452      | **-4**  |

Table: Normalized perplexities for each model on the test books.

Extension
=========

For the extension, we chose to employ our n-gram model in the service of an NLP application. We created a Sublime Text plugin, BigramsCompleteMe^[https://github.com/fredcallaway/BigramsCompleteMe], which sorts autocompletion results based on the probability given the previously typed word. (See the linked Github page for a demonstration and installation instructions). The plugin is designed to be used for editing text files, including plain text, \LaTeX, and markdown (which this document was written in).

The utility of the tool derives from the improved `insert_best_completion` performance, with the goal of increasing typing speed by saving the writer from typing long words and phrases repeatedly. It is not clear how Sublime Text's default sorting algorithm works, but it is some combination of frequency and recency. This is highly effective for editing source code which has a small number of unique words that tend to be locally clustered. However it leaves much to be desired for natural language text completion. Because

At present, the plugin only uses the `Distribution` class because at each completion, only the bigrams beginning with one word are relevant. Calling the full model causes noticeable delay as the file approaches 5000 words. It is difficult to know when a user has deleted a word, thus retraining upon every completion is the only straightforward and reliable option. However, future versions of the plugin could use pre-trained `BigramModels`, which could come packaged with the plugin or be user-generated with a Sublime Text command.
