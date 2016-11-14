# Project 3 Proposal

### Model

We plan to perform sequence tagging using **Hidden Markov Models**. Recognizing named entities will essentially be accomplished by using the Viterbi algorithm to calculate the probability of part of speech sequences in training, focusing on the probabilities that a particular token or sequence of tokens is indeed a named entity (and certain type of named entity) based upon its part-of-speech tag and position in the sequence.

#### Algorithm Key Points

*Train the Model*

- Extract triplets of form (sentence, POS tags, named entity tags) from the training data
- Get counts of bigrams of named entity tags and each POS tag given a NE tag
- Calculate probabilities using said counts and store in two 2x2 matrices:
    - Matrix 1 (transition probabilities): rows=NE tags, columns = NE tags, intersections = P(ei|ei-1)
    - Matrix 2 (emission probabilities): rows=NE tags, columns = POS tags, intersections = P(ti|ei)
- Smooth emission probabilities matrix
- Identify floor probability for what constitutes a named entity by applying gradient descent to a validation set

*Apply Model to Test Data*


- For each sentence in the test data, calculate the probability of each word being the tagged as beginning or part of each type of named entity by applying the Viterbi algorithm to the sentence using an overall probability calculated by the sum of negative log of transition probabilities and emission probabilities for a sentence.
- Named entities are those whose probabilities exceed the floor value determined at the end of training for each type of entity
- Store the indices of each classified/recognized named entity and write to file

### Baseline
As a baseline, we have implemented a model that tags all entities in the test set that were seen in the training set. This is simply a lookup table. At present, we do not prevent the extracted entities from overlapping; however this is something we hope to do for the baseline that we compare our model to. Our current baseline has an accuracy of 0.59882 on the test set.


### Extension(s)

We hope to implement a **Maximum Entropy Markov Model** for our extension as well as perhaps testing the changes caused by employing **n-gram models larger than bigram** (i.e. trigram, 4-gram). For the MEMM, we will extract features using the spacy python library. We will begin with identical features as to the ones we used for word sense disambiguation, and then refine as needed. We expect that dependency relationships will be especially valuable for NER. For example, being the subject with most verbs other than "to be" would be a good indication of a named entity.
