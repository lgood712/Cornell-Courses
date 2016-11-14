# Named Entity Recognition

Our team implemented two sequence tagging methods. One is the Conditional Random Field method, implemented using a library, and the other is a Hidden Markov Model, implemented from scratch.

## 1. Sequence Tagging Model

### Pre-Processing
When parsing the data from both the training and test sets, we created lists of triplet tuples of the following form ([tokens], [POS tags], [NE tags/indices]). Each triplet represents a sentence from the data, and allows for easy accesibility to different aspects of sentences as we progressed in training and testing our named entity recognizers.

### Implementation Details
We use a [python wrapper](https://github.com/tpeng/python-crfsuite) around the [crfsuite](http://www.chokkan.org/software/crfsuite/) package. We chose this library because of its speed and quality of documentation. We extract features using [spaCy](https://spacy.io) for the same reasons. SpaCy allows us to use rich linguistic features including syntactic dependencies and brown clusters. We also extract collocation features and orthographic features for the target word (e.g. "is in Title Case").

### Introspection
The python package provides handly tools for understanding how the model works. We employed these tools to discover highly weighted assocations.

#### Transitions
```
Top likely transitions           Top unlikely transitions
-------------------------        --------------------------
B-ORG  -> I-ORG   7.107347       O      -> I-ORG   -2.822595
B-PER  -> I-PER   6.516069       O      -> I-LOC   -2.615694
I-ORG  -> I-ORG   6.065302       O      -> I-MISC  -2.393483
B-MISC -> I-MISC  5.509491       O      -> I-PER   -2.300188
B-LOC  -> I-LOC   5.170477       B-PER  -> B-PER   -2.029059
I-MISC -> I-MISC  4.874122       I-PER  -> B-PER   -1.873423
I-LOC  -> I-LOC   4.409071       B-ORG  -> B-ORG   -1.712804
I-PER  -> I-PER   3.362128       I-ORG  -> B-ORG   -1.608914
O      -> B-MISC  1.771754       B-LOC  -> B-PER   -1.486257
O      -> B-PER   1.709055       B-PER  -> B-ORG   -1.465346
O      -> O       1.318192       B-ORG  -> B-PER   -1.361675
O      -> B-LOC   1.164400       I-PER  -> B-ORG   -1.330654
O      -> B-ORG   0.782741       I-ORG  -> B-PER   -1.294495
B-LOC  -> B-MISC  0.671527       I-PER  -> B-LOC   -1.187749
B-LOC  -> O       0.306114       I-ORG  -> B-LOC   -1.049057
```

#### Features
```
6.709668 O      _col_0_lower__punct_
5.839274 O      _col_0_lower__number_
5.156731 O      _col_0_lower__numberpunctnumberpunctnumber_
5.053550 O      _col_0_lower__numberpunctnumber_
4.759974 B-LOC  _col_0_lower__upunctspunct_
4.202116 O      _col_0_lower__punctnumber_
3.918262 O      _col_0_lower__numberpunct_
3.822815 B-LOC  _col_0_cluster_1446_
3.814060 O      _col_0_lower__punctpunct_
3.688582 B-PER  _col_-1_lower__numberpunct_
3.628325 B-LOC  _col_0_cluster_3494_
3.425758 O      _col_0_cluster_81_
3.347058 B-ORG  _col_0_lower__ajax_
3.324743 B-LOC  _col_0_lower__england_
3.223579 O      _title_False_
3.209317 O      _col_0_cluster_874_
3.115732 B-MISC _col_0_cluster_151_
2.829683 O      _col_0_cluster_209_
2.808394 O      _col_0_lower__numberth_
2.783777 B-LOC  _col_1_lower__numberpunctnumberpunctnumber_
```

#### Experiments
*Describe the motivations and methodology of the ex-
periments that you ran. Clearly state what were your hypotheses
and what were your expectations. 

#### Results
*Summarize the performance of your system and any varia-
tions that you experimented with on both the training/validation and
test dataset. Note that you have to compare your own sys-
tem to at least one other non-trivial baseline system. Put
the results into clearly labeled tables or diagrams and include your
observations and analysis. An error analysis is required { e.g. what
sorts of errors occurred, why? When did the system work well, when
did it fail and any ideas as to why? How might you improve the
system?*

#### Competition Score

Our team name is *jake_sousa_LIB*, in honor of our dear, dear friend. Our best Kaggle performance was achieved using our CRF library implementation with a score of 0.88140.

![Kaggle Screenshot](kaggle_screenshot.JPG)

## 2. Extensions

A Hidden Markov Model implementation was our primary extension. Two secondary extensions were actually added to the HMM implementation (extensions of extensions, if you will). They were the inclusion of a POS tag emission matrix and the ability to employ multiple forms of smoothing on our emission matrices.

**Hidden Markov Model**

Using training data, build sparse matrices:
- Matrix 1 (transition probabilities) 
   - Rows = NE tags, Columns = NE tags
   - Intersections = $P(e_i|e_{i-1})$, the probability of NE tag at i given the NE tag at i-1

- Matrix 2 (word emission probabilities)
   - Rows = NE tags, Columns = words
   - Intersections = $P(w_i|e_i)$, the probability of word at i given the NE tag at i

For both of the above matrices, we started by going through the training data and recording counts of the different occurrences in the matrices (i.e. count of "Luke" in coincidence with "B-PER" for Matrix 2) We then smooth the counts in the emission matrix using either Good-Turing or Add-*k* smoothing, before translating those counts into probabilities.

Using the calculated probabilities from the matrices, we run each parsed test sentence through a function which runs the Viterbi algorithm and returns the most likely path (sequence of NE tags) for that sentence. The indices of all named entity tags in the predicted path are recorded in a dictionary. An additional row in the transition matrix is reserved for counts of first NER tags in each sentence and is used to calculate the probability of each NER tag appearing at the begging of a sentence when running the Viterbi algorithm.

**Extensions of the Extension**

*Part-of-speech emission probabilities*

The inclusion of part-of-speech emission probabilities was meant to boost the performance of Viterbi by providing an additional element to influence the selection of a path.

- Matrix 3 (POS emission probabilities)
   - Rows = NE tags, Columns = POS tags
   - Intersections = $P(pos_i|e_i)$, the probability of the POS tag at i given the NE tag at i
   
*Different smoothing methods*

We used implementations of Good-Turing and Add-k smoothing from previous projects to allow us to test and select the smoothing that led to the best performance. Ultimately, we use Good-Turing smoothing on the emission matrices and Add-1e67 on the transition matrix to account for zero probabilities, which for all intents and purposes gives those paths a zero probability without sacrificing the ability to calculate surprisals for those transitions.



type | F1 |   PREC | REC
-----|----|------|------
LOC |  0.708 |0.632 |0.804
MISC | 0.340 |0.235 |0.621
PER |  0.804 |0.869 |0.748
ORG |  0.676 |0.635 |0.723
avg |  0.632 |0.593 |0.724

type | F1    | PREC  | REC
-----|-------|-------|------
LOC  | 0.820 | 0.775 | 0.871
MISC | 0.670 | 0.737 | 0.613
PER  | 0.867 | 0.853 | 0.882
ORG  | 0.787 | 0.780 | 0.793
avg  | 0.786 | 0.786 | 0.790

## Individual Member Contribution

Frederick Laws Callaway
- Implemented CRF library approach
- Wrote functions for calculating precision, recall, and F1 from a validation set
- Wrote functions for introspection

Luke Nathaniel Walter Goodman
- Wrote functions for parsing data and getting validation sets
- Implemented baseline approach (known entities approach)
- Implemented HMM approach and extensions
-
