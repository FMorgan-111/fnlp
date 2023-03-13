from typing import List, Tuple

import nltk, inspect, sys, hashlib

from nltk.corpus import brown

import math

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist, LidstoneProbDist
from nltk.probability import ELEProbDist

from nltk.tag import map_tag
import numpy as np

from adrive2 import trim_and_warn

assert map_tag('brown', 'universal', 'NR-TL') == 'NOUN', '''
Brown-to-Universal POS tag map is out of date.'''


class HMM:
    def __init__(self, train_data: List[List[Tuple[str, str]]]) -> None:
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        """
        self.train_data = train_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []
        self.viterbi = []
        self.backpointer = []

    # Q1.1

    # Compute emission model using ConditionalProbDist with
    # a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with
    #    +0.001 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self,
                       train_data: List[List[Tuple[str, str]]]) -> \
                          Tuple[ConditionalProbDist, List[str]]:
        """
        Compute an emission model based on labelled training data.
        Don't forget to lowercase the observation otherwise it mismatches
        the test data.

        :param train_data: The training dataset, a list of sentences with tags
        :return: The emission probability distribution and a list of the states
        """
        # TODO prepare data
        lower_data_list=[]
        for sentence in train_data:
            for words,tag in sentence:
                lower_data_list.append([words.lower(),tag])
        


        # Don't forget to lowercase the observation,
        # otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences
       

        # TODO compute the emission model
        emission_FD=ConditionalFreqDist()
        for words,tag in lower_data_list:

            emission_FD[tag][words]+=1
        
        lidstone_estimator = lambda fd: LidstoneProbDist(fd, 0.001, fd.B() + 1)
        
        self.emission_PD = ConditionalProbDist(emission_FD,lidstone_estimator)
        # collect the states and sort them.
        # Sorting helps with consistent tie breaking in Viterbi.
      

        self.states = sorted(set(tag for (_, tag) in lower_data_list))


        return self.emission_PD, self.states
  




    # Q1.1

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self, state: str, word: str) -> float:
        """
        The log of the estimated probability of emitting a word from a state.

        If you use the math library to compute a log base 2, make sure to
        use math.log(p,2) (rather than math.log2(p))

        :param state: the state name
        :param word: the word
        :return: log base 2 of the estimated emission probability
        """
        probability =math.log(self.emission_PD[state].prob(word),2)
        return probability


    # Q1.2
    # Compute transition model using ConditionalProbDist with the same
    #  estimator as above (but without the extra bin)
    # See comments for emission_model above for details on the estimator.
    def transition_model(self,
                         train_data: List[List[Tuple[str,str]]]) -> ConditionalProbDist:
        """
        Compute a transition model using a ConditionalProbDist based on
          labelled data.

        :param train_data: The training dataset, a list of sentences with tags
        :return: The transition probability distribution
        """
        
        # TODO: prepare the data
        tag_data = []
        processed_data=[]
        final_list=[]
        for sentences in train_data:
            for (_,tag) in sentences:
               tag_data.append(tag)# to get every tag
            processed_data.append(('<s>',tag_data[0])) # make a tuple with the beginning of <s>
            for i in range (len(tag_data)-1):
                processed_data.append((tag_data[i],tag_data[i+1]))
            processed_data.append((tag_data[-1],'</s>'))# make a tuple with the end of </s>
            final_list.append(processed_data)
            tag_data=[] #clear elements in tag_data and processed_data
            processed_data=[]
        
            
            
       
               
                


        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL <s> and the END SYMBOL </s>
        

        # TODO compute the transition model

        transition_FD = ConditionalFreqDist()
        for tag_list in final_list:
            for conditions,observations in tag_list:


                transition_FD[conditions][observations]+=1
        lidstone_estimator = lambda fd: LidstoneProbDist(fd, 0.001, fd.B())
        self.transition_PD = ConditionalProbDist(transition_FD,lidstone_estimator)
        
        return self.transition_PD

    # Q1.2
    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self, state1: str, state2: str) -> float:
        """
        The log of the estimated probability of a transition from one state to another

        If you use the math library to compute a log base 2, make sure to
        use math.log(p,2) (rather than math.log2(p))

        :param state1: the first state name
        :param state2: the second state name
        :return: log base 2 of the estimated transition probability
        """
       
        return math.log(self.transition_PD[state1].prob(state2),2)

    # Train the HMM
    def train(self) -> None:
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part 2: Implementing the Viterbi algorithm.

    # Q2.1
    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag and the total number of observations.
    def initialise(self, observation: str, number_of_observations: int) -> None:
        """
        Initialise data structures self.viterbi and self.backpointer for
        tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :param number_of_observations: the number of observations
        """
       
        # Initialise step 0 of viterbi, including
        #  transition from <s> to observation
        # use costs (- log-base-2 probabilities)
        # TODO
        backpointer = np.zeros(len(self.states),number_of_observations, dtype=int) # Initialise a 2D-array for backpointer
        viterbi = np.zeros(len(self.states),number_of_observations) # initial the trellis of viterbi with a 2D-array to represent the path probability matrix
        for state in range(len(self.states)):
            emission_cost=-math.log(self.elprob(state,observation),2)
            transition_cost=-math.log(self.tlprob('<s>',state))
            viterbi[state,:]=emission_cost+transition_cost
        



       
        

    # Q2.1
    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42
    def get_viterbi_value(self, state: str, step: int) -> float:
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :return: The value (a cost) for state as of step
        """
        return self.viterbi[state][step]

        

        return 

    # Q2.1
    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state: str, step: int) -> str:
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :param step: The (0-origin) number of a step:  if negative,
          counting backwards from the end, i.e. -1 means the last step
        :return: The state name to go back to at step-1
        """
      
        return self.backpointer[state][step]

    # Q2.2
    # Tag a new sentence using the trained model and already initialised
    # data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer data structures.
    # Describe your implementation with comments.
    def tag(self, observations: List[str]) -> List[str]:
        """
        Tag a new sentence using the trained model and already initialised
        data structures.

        :param observations: List of words (a sentence) to be tagged
        :return: List of tags corresponding to each word of the input
        """
       
        tags = []
        
        for t in range(2,len(observations)): # fixme to iterate over steps
            for s in range(len(self.states)): # fixme to iterate over states


                pass # fixme to update the viterbi and backpointer data structures
                #  Use costs, not probabilities
                # If the value of the backpointer is not uniquely determined
                # because there are two options with the same cost,
                # break the tie by storing the backpointer to the first state
                # (according to the order of self.states)

        # TODO
        # Add a termination step with cost based solely on
        #   cost of transition to </s> , end of sentence.

        # TODO
        # Reconstruct the tag sequence using the backpointers.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.
        tags = ... # fixme

        return tags

    def tag_sentence(self, sentence: List[str]) -> List[str]:
        """
        Initialise the HMM, lower case and tag a sentence. Returns a list of tags.
        :param sentence: the sentence
        """
        raise NotImplementedError("HMM.tag_sentence")
        return ... # fixme



def answer_question_2_3() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :return: incorrectly tagged sequence, correctly tagged sequence and
    your answer [max 75 words not including the sequences]
    """
    raise NotImplementedError('answer_question_2_3')

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = 'fixme'
    correct_sequence = 'fixme'
    # Why do you think the tagger tagged this example incorrectly?
    answer = inspect.cleandoc(""" your answer""")

    return tagged_sequence, correct_sequence, answer

# Q3.1
def hard_em(labeled_data: List[List[Tuple[str, str]]],
            unlabeled_data: List[List[str]], k: int) -> HMM:
    """
    Run k iterations of hard EM on the labeled and unlabeled data.
    Follow the pseudo-code in the coursework instructions.

    :param labeled_data:
    :param unlabeled_data:
    :param k: number of iterations
    :return: HMM model trained with hard EM.
    """
    raise NotImplementedError()
    return ... # fix me


# Q3.2
def answer_question3_2():
    """
    Sentence:  In    fact  he    seemed   delighted  to  get   rid  of  them   .
    Gold POS:  ADP   NOUN  PRON  VERB     VERB      PRT  VERB  ADJ  ADP  PRON  .
    T_0     :  PRON  VERB  NUM    ADP     ADJ       PRT  VERB  NUM  ADP  PRON  .
    T_3     :  PRON  VERB  PRON  VERB     ADJ       PRT  VERB  NUM  ADP  NOUN  .

    1) T_0 erroneously tagged "he" as "NUM" and T_3 correctly identifies it as "PRON".
        Speculate why additional unlabeled data might have helped in that case.
        Refer to the training data (inspect the 20 sentences!).
    2) Where does T_3 mislabel a word but T_0 is correct? Why do you think hard
       EM went wrong in that case?

    :rtype: str
    :return: your answer [max 150 words]
    """
    raise NotImplementedError('answer_question_3_2')

    return inspect.cleandoc("""your answer""")


def answer_question_4_1() -> str:
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 100 words]
    """
    raise NotImplementedError('answer_question_4_1')

    return inspect.cleandoc("""your answer""")


def answer_question_4_2() -> str:
    """
    Suppose you have a hand-crafted probabilistic context-free grammar
    and lexicon that has 100% coverage on constructions but less than
    100% lexical coverage. How could you use a pre-trained POS tagger
    to ensure that the grammar produces a parse for any well-formed
    sentence, even when the lexicon doesn’t contain some of the words
    within that sentence?

    Will your approach always do as well as or better than the original
    parser on its own?  Why or why not?

    :rtype: str
    :return: your answer [max 150 words]
    """
    raise NotImplementedError('answer_question4_2')

    return inspect.cleandoc("""your answer""")


def answer_question_4_3() -> str:
    """
    Essay question on noisy channel model; see coursework instructions.

    :rtype: str
    :return: your answer [max 400 words]
    """
    raise NotImplementedError('answer_question4_3')

    return inspect.cleandoc("""your answer""")



def compute_acc(hmm, test_data, print_mistakes):
    """
    Computes accuracy (0.0 - 1.0) of model on some data.
    :param hmm: the HMM
    :type hmm: HMM
    :param test_data: the data to compute accuracy on.
    :type test_data: list(list(tuple(str, str)))
    :param print_mistakes: whether to print the first 10 model mistakes
    :type print_mistakes: bool
    :return: float
    """
    # TODO: modify this to print the first 10 sentences with at
    #  least one mistake if print_mistakes = True
    correct = 0
    incorrect = 0
    for sentence in test_data:
        s = [word for (word, tag) in sentence]
        tags = hmm.tag_sentence(s)

        for ((word, gold), tag) in zip(sentence, tags):
            if tag == gold:
                correct += 1
            else:
                incorrect += 1

    return float(correct) / (correct + incorrect)


# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def answers():
    global tagged_sentences_universal, test_data_universal, \
        train_data_universal, model, test_size, train_size, ttags, \
        correct, incorrect, accuracy, \
        good_tags, bad_tags, answer2_3, answer5, answer4_2, answer4_1, \
        answer3_2, answer4_3, t0_acc, tk_acc

    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news',
                                                    tagset='universal')

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal) - test_size

    # tail test set
    test_data_universal = tagged_sentences_universal[-test_size:]  # [:test_size]
    train_data_universal = tagged_sentences_universal[:train_size]  # [test_size:]
    if hashlib.md5(''.join(map(lambda x: x[0],
                               train_data_universal[0] +
                               train_data_universal[-1] + test_data_universal[0] +
                               test_data_universal[-1])).encode(
            'utf-8')).hexdigest() != '164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect -- '
              'this should not happen, please contact a TA !!!' % (
        len(train_data_universal), len(test_data_universal)), file=sys.stderr)

    # Create instance of HMM class and initialise the training set.
    model = HMM(train_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample = model.elprob('VERB', 'is')
    if not (type(e_sample) == float and e_sample <= 0.0):
        print('elprob value (%s) must be a log probability' % e_sample,
              file=sys.stderr)

    t_sample = model.tlprob('VERB', 'VERB')
    if not (type(t_sample) == float and t_sample <= 0.0):
        print('tlprob value (%s) must be a log probability' % t_sample,
              file=sys.stderr)

    if not (type(model.states) == list and \
            len(model.states) > 0 and \
            type(model.states[0]) == str):
        print('model.states value (%s) must be a non-empty list of strings'
              % model.states, file=sys.stderr)
    else:
        print('states: %s\n' % model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s = 'the cat in the hat came back'.split()
    ttags = model.tag_sentence(s)
    print("Tagged a trial sentence:\n  %s" % list(zip(s, ttags)))

    v_sample = model.get_viterbi_value('VERB', 5)
    if not (type(v_sample) == float and 0.0 <= v_sample):
        print('viterbi value (%s) must be a cost' % v_sample, file=sys.stderr)

    b_sample = model.get_backpointer_value('VERB', 5)
    if not (type(b_sample) == str and b_sample in model.states):
        print('backpointer value (%s) must be a state name' % b_sample,
              file=sys.stderr)

    # check the model's accuracy (% correct) using the test set
    accuracy = compute_acc(model, test_data_universal, print_mistakes=True)
    print('\nTagging accuracy for test set of %s sentences: %.4f' % (test_size,
                                                                     accuracy))

    #Tag the sentence again to put the results in memory for automarker.
    model.tag_sentence(s)

    # Question 3.1
    # Set aside the first 20 sentences of the training set
    num_sentences = 20
    semi_supervised_labeled = train_data_universal[:num_sentences]  # type list(list(tuple(str, str)))
    semi_supervised_unlabeled = \
                 [[word for (word, tag) in sent] for sent in
                  train_data_universal[num_sentences:]]  # type list(list(str))
    print("Running hard EM for Q3.2. This may take a while...")
    t0 = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 0) # 0 iterations
    tk = hard_em(semi_supervised_labeled, semi_supervised_unlabeled, 3)
    print("done.")

    t0_acc = compute_acc(t0, test_data_universal, print_mistakes=False)
    tk_acc = compute_acc(tk, test_data_universal, print_mistakes=False)
    print('\nTagging accuracy of T_0: %.4f' % (t0_acc))
    print('\nTagging accuracy of T_3: %.4f' % (tk_acc))
    ########

    # Print answers
    bad_tags, good_tags, answer2_3 = answer_question_2_3()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nDiscussion of the difference:')
    print(answer2_3)
    answer3_2 = answer_question3_2()
    print("\nFor Q3.2:")
    print(answer3_2)
    answer4_1 = answer_question_4_1()
    print('\nFor Q4.1:')
    print(answer4_1)
    answer4_2 = answer_question_4_2()
    print('\nFor Q4.2:')
    print(answer4_2)
    answer4_3 = answer_question_4_3()
    print('\nFor Q4.3:')
    print(answer4_3)



if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--answers':
        import adrive2
        from autodrive_embed import run, carefulBind

        with open("userErrs.txt", "w") as errlog:
            run(globals(), answers, adrive2.a2answers, errlog)
    else:
        answers()
