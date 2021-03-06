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
chain = itertools.chain.from_iterable
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
        tagged_words = chain(train_data)
        data = [(tag, word.lower()) for (word, tag) in tagged_words]

        # Train the emission probilistic model
        emission_FD = ConditionalFreqDist(data)
        # Reseal the lidston function with gamma 0.01 and a proper bin number
        lidstone_PD = lambda FD: LidstoneProbDist(FD, gamma=0.01, bins=FD.B() + 1)
        self.emission_PD = ConditionalProbDist(emission_FD, lidstone_PD)
        # Store the tags as states
        self.states = emission_FD.conditions()

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
        
        # Padding sentenses with <s> at beginning and </s> at end
        padded = [[('<s>', '<s>')] + s + [('</s>', '</s>')] for s  in train_data]

        # Reform the data into list[tuple(tag_(i),tag_(i+1))]
        data = chain([[(s[i][1], s[i+1][1]) \
                               for i in range(len(s)-1)] for s in padded])

        # Compute the transition model
        # Reseal the lidston function with gamma 0.01 and a proper bin number
        lidstone_PD = lambda FD: LidstoneProbDist(FD, gamma=0.01, bins=FD.B() + 1)
        transition_FD = ConditionalFreqDist(data)
        
        # Store the trainned conditinoal probabilistic distribution model
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
        
        # If the model have not trained yet, train transition model first
        if self.transition_PD == None:
            self.transition_model(self.train_data)

        # Calculate the log probability of a transition from state1 to state2
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
        
        """
        The self.viterbi is a chart of structure list[list[float]] looks like
        __________________________
        |    | o0 | o1 | o2 | ...
        | q1 |cost|cost|cost| ...
        | q2 |cost|cost|cost| ...
        | q3 |cost|cost|cost| ...
         ...  ...  ...  ...   ...
         o_n represents the nth step, also the nth input word count from 0.
         q_m represents the mth state in the self.states.
         'cost' is a float represent the negative log likehood cost
         Thus according to the position of the 'cost' we can know the lowest cost 
         at step n while q_m is its tag.
         So the cell q2o2 represents the lowest possible cost if the 2th word is 
         the state q2
         
         The self.backpointer is a chart of structure list[list[int]] looks like
        __________________________
        |    | o0 | o1 | o2 | ...
        | q1 |back|back|back| ...
        | q2 |back|back|back| ...
        | q3 |back|back|back| ...
         ...  ...  ...  ...   ...
         o_n represents the nth step, also the nth input word count from 0.
         q_m represents the mth state in the self.states.
         'back' is an integer represents the index in self.states
         The 'back' on cell q_mo_n represents the lowest cost path to q_mo_n is from 
         q_(back)o_(n-1)
         Also, q_(back) means the 'back'th state in self.states
        """
                
        # Initialise viterbi and backpointer matrix each time
        self.viterbi = []
        self.backpointer = []
        
        # The first word
        o = observation.lower()
        
        # Update the viterbi matrix, the cost from <s> to the first word
        # Transmission cost add emission cost
        self.viterbi.append([-(self.tlprob('<s>', p) + self.elprob(p, o)) for p in self.states])

        # -1 in backpointer matrix represents <s>, the first column are all point to <s>, so fill -1
        self.backpointer.append([-1 for i in self.states])

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

        # Transfer the observations into lowercase
        o = [word.lower() for word in observations]
        tags = []   # Initial the tag list
        m = len(self.states)    # number of states recorded in m

        step = 0
        for t in o: # iterate over steps, from step 1 to the last step
            # Initial the lowest cost as a very large value
            viterbi = [999999999] * m    # new column in viterbi matrix
            backpointer = [-1] * m       # new column in backpointer matrix
            
            for s in self.states:   # iterate over states, to find the best current state
                for ls in self.states:  # for each current possible state, find the best last state
                    # cost is the sum of last step cost, transmission cost from last state and the emission cost
                    cost = -(self.tlprob(ls, s) + self.elprob(s, t)) + self.get_viterbi_value(ls, step)
                    # update the viterbi and backpointer matrix if the new cost is smaller
                    if cost < viterbi[self.states.index(s)]:
                        viterbi[self.states.index(s)] = cost
                        backpointer[self.states.index(s)] = self.states.index(ls)
            step += 1   # do next step/word
            # append the new columns
            self.viterbi.append(viterbi)
            self.backpointer.append(backpointer)

        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.
        terminate = [0] * m   # Initial the terminate step
        # Count from 0, update the cost
        idx = 0
        for ls in self.states:
            # add the transmission cost from the final state to </s>
            terminate[idx] = self.get_viterbi_value(ls, step) - self.tlprob(ls, '</s>')
            idx += 1

        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.

        # Find the state with minimum cost in the terminate step
        backpointer = self.states[terminate.index(min(terminate))]
        # Retrace the tags from the minimum cost in terminate
        while backpointer != '<s>':
            tags.append(backpointer)
            backpointer = self.get_backpointer_value(backpointer, step)
            step -= 1
        # Reverse the tags so it has a normal sequence
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
        return self.viterbi[step][self.states.index(state)]

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
    tagged_sequence = [('``', '.'), ('My', 'DET'), ('taste', 'NOUN'), ('is', 'VERB'), ('gaudy', 'ADV'), ('.', '.')]
    correct_sequence = [('``', '.'), ('My', 'DET'), ('taste', 'NOUN'), ('is', 'VERB'), ('gaudy', 'ADJ'), ('.', '.')]
    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("""The HMM model can only capture 2-word history, not long-range dependencies. 'gaudy' is for 'taste', but HMM model only knows it follows a VERB, so tags it as ADV rather than ADJ. Because ADV is more likely follows a VERB, and 'gaudy' has similar cost being ADJ or ADV.""")[0:280]

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

    return inspect.cleandoc("""When no global ambiguities and no unkonw words, the original parser only have 1 valid output, we directly use the output. If there are global ambiguities, the original parser has multiple valid results, use the pre-trained POS tagger to find the most likely one. And if there are unknown words, use the pre-trained tagger to find the most likely tag of that word depends on the transition probabilitiy. So it always better or as well as the original parser.""")[0:500]


def answer_question6():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    # raise NotImplementedError('answer_question6')

    return inspect.cleandoc("""If we use the original tagset, the accuracy on the test set will be much lower. Because with more tags, each tag/word pair has less ovservations, so we have few confidence level on the probability model. And more tags depends on long-range effects, but HMM nodel only catch 2 word history, so they are more errors. And using large complex tagset the annotor is more likely to make errors. Thus the overall accuracy will be lower.""")[0:500]


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

    test_data_universal = tagged_sentences_universal[-test_size:]
    train_data_universal = tagged_sentences_universal[:train_size]

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
                correct += 1
            else:
                incorrect += 1
                wrong = True
        if wrong and idx < 10:
            idx += 1
            gold_sents.append(sentence)
            origin_sent = [word for (word, tag) in sentence]
            tagged_sents.append(list(zip(origin_sent, tags)))
    print(gold_sents)
    print(tagged_sents)

    # Calculate the accuracy
    accuracy = correct / (correct + incorrect)
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
