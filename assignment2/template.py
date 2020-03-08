import inspect, sys, hashlib

# Hack around a warning message deep inside scikit learn, loaded by nltk :-(
#  Modelled on https://stackoverflow.com/a/25067818
import warnings
with warnings.catch_warnings(record=True) as w:
    save_filters=warnings.filters
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    import nltk
    warnings.filters=save_filters
try:
    nltk
except NameError:
    # didn't load, produce the warning
    import nltk

from nltk.corpus import brown
from nltk.tag import map_tag, tagset_mapping
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist

from nltk.probability import LidstoneProbDist
import itertools
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if map_tag('brown', 'universal', 'NR-TL') != 'NOUN':
    # Out-of-date tagset, we add a few that we need
    tm=tagset_mapping('en-brown','universal')
    tm['NR-TL']=tm['NR-TL-HL']='NOUN'

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.01 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """
        # raise NotImplementedError('HMM.emission_model')

        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences

        # Repack the train_data to list(tuple(tag, lowercase_word)) format
        tagged_words = itertools.chain.from_iterable(train_data)
        data = [(tag, word.lower()) for (word, tag) in tagged_words]

        # Train the emission probilistic model
        emission_FD = ConditionalFreqDist(data)
        lidstone_PD = lambda FD: LidstoneProbDist(FD, gamma=0.01, bins=FD.B())
        self.emission_PD = ConditionalProbDist(emission_FD, lidstone_PD)
        self.states = emission_FD.conditions() # tags

        return self.emission_PD, self.states

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self,state,word):
        """
        The log of the estimated probability of emitting a word from a state

        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """
        # Avoid repeating training
        if self.emission_PD == None:
            self.emission_model(self.train_data)

        # raise NotImplementedError('HMM.elprob')

        # Calculate the log probability of a word with given tag/state, base 2
        return self.emission_PD[state].logprob(word)

    # Compute transition model using ConditionalProbDist with a LidstonelprobDist estimator.
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
        # raise NotImplementedError('HMM.transition_model')


        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>
        padded = [[('<s>', '<s>')] + s + [('</s>', '</s>')] for s  in train_data]

        data = itertools.chain.from_iterable([[(s[i][1], s[i+1][1]) for i in range(len(s)-1)] for s in padded])

        # Compute the transition model
        lidstone_PD = lambda FD: LidstoneProbDist(FD, gamma=0.01, bins=FD.B())
        transition_FD = ConditionalFreqDist(data)
        self.transition_PD = ConditionalProbDist(transition_FD, lidstone_PD)

        return self.transition_PD

    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self,state1,state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        # raise NotImplementedError('HMM.tlprob')
        if self.transition_PD == None:
            self.transition_model(self.train_data)

        return self.transition_PD[state1].logprob(state2)

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag
    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        """
        # raise NotImplementedError('HMM.initialise')
        # Initialise step 0 of viterbi, including
        #  transition from <s> to observation
        # use costs (-log-base-2 probabilities)
        # if the backpointer is 0 means this word is at the beginning
        o = observation.lower()
        self.viterbi = []
        self.viterbi.append([-(self.tlprob('<s>', o) + self.elprob(p, o)) for p in self.states])

        self.backpointer = []
        self.backpointer.append((-1 for i in range(len(self.states))))


    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer datastructures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
        # raise NotImplementedError('HMM.tag')

        o = [w.lower() for w in observations]
        tags = []

        step = 0
        for t in o: # fixme to iterate over steps
            viterbi = [999999999] * len(self.states)
            backpointer = [-1] * len(self.states)
            
            for s in self.states: # fixme to iterate over states
                for ls in self.states:
                    cost = -(self.tlprob(ls, s) + self.elprob(s, t)) \
                    + self.get_viterbi_value(ls, step)
                    if cost < viterbi[self.states.index(s)]:
                        viterbi[self.states.index(s)] = cost
                        backpointer[self.states.index(s)] = self.states.index(ls)
            step += 1
            self.viterbi.append(viterbi)
            self.backpointer.append(backpointer)

        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.
        ter = [0] * len(self.states)
        minCost = 999999
        for ls in self.states:
            newCost = self.get_viterbi_value(ls, step) - self.tlprob(ls, '</s>')
            if newCost < minCost:
                minCost = newCost
                newBackpointer = self.states.index(ls)
        self.viterbi.append([minCost] * len(self.states))
        self.backpointer.append([newBackpointer] * len(self.states))
                       

        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.
        step += 1
        backpointer = self.get_backpointer_value('</s>', step)
        while backpointer != '<s>':
            step -= 1
            tags.append(backpointer)
            backpointer = self.get_backpointer_value(backpointer, step)
        tags.reverse()

        return tags

    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42
    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        # raise NotImplementedError('HMM.get_viterbi_value')
        nstep = self.states.index(state)
        vit = self.viterbi[step]
        return vit[nstep]

    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :type step: int
        :return: The state name to go back to at step-1
        :rtype: str
        """
        # raise NotImplementedError('HMM.get_backpointer_value')

        if step == 0 or step == -len(self.viterbi):
            return '<s>'
        if state == '</s>' and (step == len(self.viterbi) - 1 or step == -1):
            return self.states[self.backpointer[step][0]]
        return self.states[self.backpointer[step][self.states.index(state)]]

def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: your answer [max 280 chars]
    """
    # raise NotImplementedError('answer_question4b')

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = [[('Tooling', 'X'), ('through', 'ADP'), ('Sydney', 'NOUN'), ('on', 'ADP'), ('his', 'DET'), ('way', 'NOUN'), ('to', 'ADP'), ('race', 'NOUN'), ('in', 'ADP'), ('the', 'DET'), ('New', 'ADJ'), ('Zealand', 'X'), ('Grand', 'X'), ('Prix', 'X'), (',', '.'), ("Britain's", 'X'), ('balding', 'X'), ('Ace', 'X'), ('Driver', 'X'), ('Stirling', 'X'), ('Moss', 'X'), (',', '.'), ('31', 'NUM'), (',', '.'), ('all', 'PRT'), ('but', 'CONJ'), ('smothered', 'ADV'), ('himself', 'PRON'), ('in', 'ADP'), ('his', 'DET'), ('own', 'ADJ'), ('exhaust', 'NOUN'), ('of', 'ADP'), ('self-crimination', 'NUM'), ('.', '.')], [('``', '.'), ('My', 'DET'), ('taste', 'NOUN'), ('is', 'VERB'), ('gaudy', 'ADV'), ('.', '.')], [("I'm", 'PRT'), ('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'CONJ'), ('racing', 'ADJ'), ('cars', 'NOUN'), ('.', '.')], [("I'm", 'X'), ('ruddy', 'X'), ('lazy', 'X'), (',', '.'), ('and', 'CONJ'), ("I'm", 'PRT'), ('getting', 'VERB'), ('on', 'ADP'), ('in', 'ADP'), ('years', 'NOUN'), ('.', '.')], [('It', 'PRON'), ('gets', 'VERB'), ('so', 'ADV'), ('frustrating', 'ADV'), (',', '.'), ('but', 'CONJ'), ('then', 'ADV'), ('again', 'ADV'), ('I', 'PRON'), ("don't", 'VERB'), ('know', 'VERB'), ('what', 'DET'), ('I', 'PRON'), ('could', 'VERB'), ('do', 'VERB'), ('if', 'ADP'), ('I', 'PRON'), ('gave', 'VERB'), ('up', 'PRT'), ('racing', 'VERB'), ("''", '.'), ('.', '.')], [('Has', 'X'), ('Moss', 'X'), ('no', 'X'), ('stirling', 'X'), ('virtues', 'X'), ('?', '.'), ('?', '.')], [('One', 'NUM'), ('of', 'ADP'), ('Nikita', 'X'), ("Khrushchev's", 'X'), ('most', 'X'), ('enthusiastic', 'X'), ('eulogizers', 'X'), (',', '.'), ('the', 'DET'), ("U.S.S.R.'s", 'X'), ('daily', 'X'), ('Izvestia', 'X'), (',', '.'), ('enterprisingly', 'X'), ('interviewed', 'X'), ('Red-prone', 'X'), ('Comedian', 'X'), ('Charlie', 'X'), ('Chaplin', 'X'), ('at', 'ADP'), ('his', 'DET'), ('Swiss', 'ADJ'), ('villa', 'NOUN'), (',', '.'), ('where', 'ADV'), ('he', 'PRON'), ('has', 'VERB'), ('been', 'VERB'), ('in', 'ADP'), ('self-exile', 'X'), ('since', 'X'), ('1952', 'X'), ('.', '.')], [('Chaplin', 'X'), (',', '.'), ('71', 'NUM'), (',', '.'), ('who', 'PRON'), ('met', 'VERB'), ('K.', 'NOUN'), ('when', 'ADV'), ('the', 'DET'), ('Soviet', 'NOUN'), ('boss', 'NOUN'), ('visited', 'VERB'), ('England', 'NOUN'), ('in', 'ADP'), ('1956', 'NUM'), (',', '.'), ('confided', 'ADV'), ('that', 'ADP'), ('he', 'PRON'), ('hopes', 'VERB'), ('to', 'PRT'), ('visit', 'VERB'), ('Russia', 'NOUN'), ('some', 'DET'), ('time', 'NOUN'), ('this', 'DET'), ('summer', 'NOUN'), ('because', 'ADV'), ('``', '.'), ('I', 'PRON'), ('have', 'VERB'), ('marveled', 'ADV'), ('at', 'ADP'), ('your', 'DET'), ('grandiose', 'ADJ'), ('experiment', 'NOUN'), ('and', 'CONJ'), ('I', 'PRON'), ('believe', 'VERB'), ('in', 'ADP'), ('your', 'DET'), ('future', 'NOUN'), ("''", '.'), ('.', '.')], [('Then', 'X'), ('Charlie', 'X'), ('spooned', 'X'), ('out', 'PRT'), ('some', 'DET'), ('quick', 'ADJ'), ('impressions', 'NOUN'), ('of', 'ADP'), ('the', 'DET'), ('Nikita', 'NOUN'), ('he', 'PRON'), ('had', 'VERB'), ('glimpsed', 'ADV'), (':', '.'), ('``', '.'), ('I', 'PRON'), ('was', 'VERB'), ('captivated', 'ADV'), ('by', 'ADP'), ('his', 'DET'), ('humor', 'NOUN'), (',', '.'), ('frankness', 'NOUN'), ('and', 'CONJ'), ('good', 'ADJ'), ('nature', 'NOUN'), ('and', 'CONJ'), ('by', 'ADP'), ('his', 'DET'), ('kind', 'NOUN'), (',', '.'), ('strong', 'ADJ'), ('and', 'CONJ'), ('somewhat', 'ADV'), ('sly', 'ADJ'), ('face', 'NOUN'), ("''", '.'), ('.', '.')], [('G.', 'NOUN'), ('David', 'NOUN'), ('Thompson', 'NOUN'), ('is', 'VERB'), ('one', 'NUM'), ('of', 'ADP'), ('those', 'DET'), ('names', 'NOUN'), ('known', 'VERB'), ('to', 'ADP'), ('the', 'DET'), ('stewards', 'NOUN'), ('of', 'ADP'), ('transatlantic', 'DET'), ('jetliners', 'NOUN'), ('and', 'CONJ'), ('to', 'ADP'), ('doormen', 'NOUN'), ('in', 'ADP'), ("Europe's", 'DET'), ('best', 'ADJ'), ('hotels', 'NOUN'), (',', '.'), ('but', 'CONJ'), ('he', 'PRON'), ('is', 'VERB'), ('somewhat', 'ADV'), ('of', 'ADP'), ('an', 'DET'), ('enigma', 'NOUN'), ('to', 'ADP'), ('most', 'ADJ'), ('people', 'NOUN'), ('in', 'ADP'), ('his', 'DET'), ('own', 'ADJ'), ('home', 'NOUN'), ('town', 'NOUN'), ('of', 'ADP'), ('Pittsburgh', 'NOUN'), ('.', '.')]]
    correct_sequence = [[('Tooling', 'VERB'), ('through', 'ADP'), ('Sydney', 'NOUN'), ('on', 'ADP'), ('his', 'DET'), ('way', 'NOUN'), ('to', 'PRT'), ('race', 'VERB'), ('in', 'ADP'), ('the', 'DET'), ('New', 'ADJ'), ('Zealand', 'NOUN'), ('Grand', 'X'), ('Prix', 'X'), (',', '.'), ("Britain's", 'NOUN'), ('balding', 'ADJ'), ('Ace', 'NOUN'), ('Driver', 'NOUN'), ('Stirling', 'NOUN'), ('Moss', 'NOUN'), (',', '.'), ('31', 'NUM'), (',', '.'), ('all', 'PRT'), ('but', 'ADP'), ('smothered', 'VERB'), ('himself', 'PRON'), ('in', 'ADP'), ('his', 'DET'), ('own', 'ADJ'), ('exhaust', 'NOUN'), ('of', 'ADP'), ('self-crimination', 'NOUN'), ('.', '.')], [('``', '.'), ('My', 'DET'), ('taste', 'NOUN'), ('is', 'VERB'), ('gaudy', 'ADJ'), ('.', '.')], [("I'm", 'PRT'), ('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'ADP'), ('racing', 'VERB'), ('cars', 'NOUN'), ('.', '.')], [("I'm", 'PRT'), ('ruddy', 'ADV'), ('lazy', 'ADJ'), (',', '.'), ('and', 'CONJ'), ("I'm", 'PRT'), ('getting', 'VERB'), ('on', 'PRT'), ('in', 'ADP'), ('years', 'NOUN'), ('.', '.')], [('It', 'PRON'), ('gets', 'VERB'), ('so', 'ADV'), ('frustrating', 'ADJ'), (',', '.'), ('but', 'CONJ'), ('then', 'ADV'), ('again', 'ADV'), ('I', 'PRON'), ("don't", 'VERB'), ('know', 'VERB'), ('what', 'DET'), ('I', 'PRON'), ('could', 'VERB'), ('do', 'VERB'), ('if', 'ADP'), ('I', 'PRON'), ('gave', 'VERB'), ('up', 'PRT'), ('racing', 'VERB'), ("''", '.'), ('.', '.')], [('Has', 'VERB'), ('Moss', 'NOUN'), ('no', 'DET'), ('stirling', 'ADJ'), ('virtues', 'NOUN'), ('?', '.'), ('?', '.')], [('One', 'NUM'), ('of', 'ADP'), ('Nikita', 'NOUN'), ("Khrushchev's", 'NOUN'), ('most', 'ADV'), ('enthusiastic', 'ADJ'), ('eulogizers', 'NOUN'), (',', '.'), ('the', 'DET'), ("U.S.S.R.'s", 'NOUN'), ('daily', 'ADJ'), ('Izvestia', 'NOUN'), (',', '.'), ('enterprisingly', 'ADV'), ('interviewed', 'VERB'), ('Red-prone', 'ADJ'), ('Comedian', 'NOUN'), ('Charlie', 'NOUN'), ('Chaplin', 'NOUN'), ('at', 'ADP'), ('his', 'DET'), ('Swiss', 'ADJ'), ('villa', 'NOUN'), (',', '.'), ('where', 'ADV'), ('he', 'PRON'), ('has', 'VERB'), ('been', 'VERB'), ('in', 'ADP'), ('self-exile', 'NOUN'), ('since', 'ADP'), ('1952', 'NUM'), ('.', '.')], [('Chaplin', 'NOUN'), (',', '.'), ('71', 'NUM'), (',', '.'), ('who', 'PRON'), ('met', 'VERB'), ('K.', 'NOUN'), ('when', 'ADV'), ('the', 'DET'), ('Soviet', 'NOUN'), ('boss', 'NOUN'), ('visited', 'VERB'), ('England', 'NOUN'), ('in', 'ADP'), ('1956', 'NUM'), (',', '.'), ('confided', 'VERB'), ('that', 'ADP'), ('he', 'PRON'), ('hopes', 'VERB'), ('to', 'PRT'), ('visit', 'VERB'), ('Russia', 'NOUN'), ('some', 'DET'), ('time', 'NOUN'), ('this', 'DET'), ('summer', 'NOUN'), ('because', 'ADP'), ('``', '.'), ('I', 'PRON'), ('have', 'VERB'), ('marveled', 'VERB'), ('at', 'ADP'), ('your', 'DET'), ('grandiose', 'ADJ'), ('experiment', 'NOUN'), ('and', 'CONJ'), ('I', 'PRON'), ('believe', 'VERB'), ('in', 'ADP'), ('your', 'DET'), ('future', 'NOUN'), ("''", '.'), ('.', '.')], [('Then', 'ADJ'), ('Charlie', 'NOUN'), ('spooned', 'VERB'), ('out', 'PRT'), ('some', 'DET'), ('quick', 'ADJ'), ('impressions', 'NOUN'), ('of', 'ADP'), ('the', 'DET'), ('Nikita', 'NOUN'), ('he', 'PRON'), ('had', 'VERB'), ('glimpsed', 'VERB'), (':', '.'), ('``', '.'), ('I', 'PRON'), ('was', 'VERB'), ('captivated', 'VERB'), ('by', 'ADP'), ('his', 'DET'), ('humor', 'NOUN'), (',', '.'), ('frankness', 'NOUN'), ('and', 'CONJ'), ('good', 'ADJ'), ('nature', 'NOUN'), ('and', 'CONJ'), ('by', 'ADP'), ('his', 'DET'), ('kind', 'ADJ'), (',', '.'), ('strong', 'ADJ'), ('and', 'CONJ'), ('somewhat', 'ADV'), ('sly', 'ADJ'), ('face', 'NOUN'), ("''", '.'), ('.', '.')], [('G.', 'NOUN'), ('David', 'NOUN'), ('Thompson', 'NOUN'), ('is', 'VERB'), ('one', 'NUM'), ('of', 'ADP'), ('those', 'DET'), ('names', 'NOUN'), ('known', 'VERB'), ('to', 'ADP'), ('the', 'DET'), ('stewards', 'NOUN'), ('of', 'ADP'), ('transatlantic', 'ADJ'), ('jetliners', 'NOUN'), ('and', 'CONJ'), ('to', 'ADP'), ('doormen', 'NOUN'), ('in', 'ADP'), ("Europe's", 'NOUN'), ('best', 'ADJ'), ('hotels', 'NOUN'), (',', '.'), ('but', 'CONJ'), ('he', 'PRON'), ('is', 'VERB'), ('somewhat', 'ADV'), ('of', 'ADP'), ('an', 'DET'), ('enigma', 'NOUN'), ('to', 'ADP'), ('most', 'ADJ'), ('people', 'NOUN'), ('in', 'ADP'), ('his', 'DET'), ('own', 'ADJ'), ('home', 'NOUN'), ('town', 'NOUN'), ('of', 'ADP'), ('Pittsburgh', 'NOUN'), ('.', '.')]]
    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("""\
    fill me in""")[0:280]

    return tagged_sequence, correct_sequence, answer

def answer_question5():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    # raise NotImplementedError('answer_question5')

    return inspect.cleandoc("""\
    fill me in""")[0:500]

def answer_question6():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    # raise NotImplementedError('answer_question6')

    return inspect.cleandoc("""\
    fill me in""")[0:500]

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5

    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal) - test_size

    test_data_universal = tagged_sentences_universal[-test_size:] # fixme
    train_data_universal = tagged_sentences_universal[:train_size] # fixme

    if hashlib.md5(''.join(map(lambda x:x[0],train_data_universal[0]+train_data_universal[-1]+test_data_universal[0]+test_data_universal[-1])).encode('utf-8')).hexdigest()!='164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect, most of your answers will be wrong hereafter!!!'%(len(train_data_universal),len(test_data_universal)),file=sys.stderr)

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample=model.elprob('VERB','is')
    if not (type(e_sample)==float and e_sample<=0.0):
        print('elprob value (%s) must be a log probability'%e_sample,file=sys.stderr)

    t_sample=model.tlprob('VERB','VERB')
    if not (type(t_sample)==float and t_sample<=0.0):
           print('tlprob value (%s) must be a log probability'%t_sample,file=sys.stderr)

    if not (type(model.states)==list and \
            len(model.states)>0 and \
            type(model.states[0])==str):
        print('model.states value (%s) must be a non-empty list of strings'%model.states,file=sys.stderr)

    print('states: %s\n'%model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s='the cat in the hat came back'.split()
    model.initialise(s[0])
    ttags = model.tag(s[1:])
    print("Tagged a trial sentence:\n  %s"%list(zip(s,ttags)))

    v_sample=model.get_viterbi_value('VERB',5)
    if not (type(v_sample)==float and 0.0<=v_sample):
           print('viterbi value (%s) must be a cost'%v_sample,file=sys.stderr)

    b_sample=model.get_backpointer_value('VERB',5)
    if not (type(b_sample)==str and b_sample in model.states):
           print('backpointer value (%s) must be a state name'%b_sample,file=sys.stderr)


    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0

    idx = 0
    gold_sents = []
    tagged_sents = []
    for sentence in test_data_universal:
        wrong = False
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s[1:])

        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                correct += 1 # fix me
            else:
                incorrect += 1 # fix me
                wrong = True
        if wrong and idx < 10:
            idx += 1
            gold_sents.append(sentence)
            origin_sent = [word for (word, tag) in sentence]
            tagged_sents.append(list(zip(origin_sent, tags)))
    # print(gold_sents)
    # print(tagged_sents)
        

    accuracy = correct / (correct + incorrect) # fix me
    print('Tagging accuracy for test set of %s sentences: %.4f'%(test_size,accuracy))

    # Print answers for 4b, 5 and 6
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nDiscussion of the difference:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])
    answer6=answer_question6()
    print('\nFor Q6:')
    print(answer6[:500])

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == '--answers':
        import adrive2_embed
        from autodrive_embed import run, carefulBind
        with open("userErrs.txt","w") as errlog:
            run(globals(),answers,adrive2_embed.a2answers,errlog)
    else:
        answers()
