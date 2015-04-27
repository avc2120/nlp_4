from __future__  import division
import nltk
import A
from collections import defaultdict
import itertools
import math as math
from nltk.align import Alignment, AlignedSent
import sys
from collections import defaultdict
from nltk.corpus import comtrans

class BerkeleyAligner():
    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        alignment = []

        l = len(align_sent.words)
        m = len(align_sent.mots)

        for j, en_word in enumerate(align_sent.words):
            
            # Initialize the maximum probability with Null token
            max_align_prob = (-sys.maxint, None)
            for i, g_word in enumerate(align_sent.mots):
                # Find out the maximum probability
                max_align_prob = max(max_align_prob,
                    (self.t[en_word][g_word]*self.q[(i+1,j+1,l,m)], i))

            # If the maximum probability is not Null token,
            # then append it to the alignment. 
            if max_align_prob[1] is not None:
                alignment.append((j, max_align_prob[1]))

        return AlignedSent(align_sent.words, align_sent.mots, alignment)

        
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def initialize(self, aligned_sents, num_iter, e_to_g):
        print 'getting counts'
        total_e = defaultdict(float)
        t = defaultdict(lambda: defaultdict(lambda: 0.0))
        counts = defaultdict(set)

        for alignSent in aligned_sents:
            english = alignSent.mots if not e_to_g else alignSent.words
            german = alignSent.words if not e_to_g else alignSent.mots
            for g_word in german:
                counts[g_word].update(english)
        for key in counts.keys():
            values = counts[key]
            for value in values:
                t[value][key] = 1.0/len(counts[key])

        q = defaultdict(float)

        for alignSent in aligned_sents:
            english = alignSent.mots if not e_to_g else alignSent.words
            german = [None] + alignSent.words if not e_to_g else [None] + alignSent.mots
            m = len(german) - 1
            l = len(english)
            initial_value = 1 / (m + 1)
            for i in range(0, m+1):
                for j in range(1, l+1):
                    q[(i,j,l,m)] = initial_value
        return t,q

    def train(self, aligned_sents, num_iter):
        print 'start train'
        t, q = self.initialize(aligned_sents, num_iter, True)  
        t1, q1 = self.initialize(aligned_sents, num_iter, False)  

        counts = defaultdict(set)
        c = defaultdict(float)
        c1 = defaultdict(float)
        total_e = defaultdict(float)

        print 'collecting counts'
        for k in range(0, num_iter):
            for alignSent in aligned_sents:
                english = alignSent.mots 
                german = [None] + alignSent.words
                m = len(german) - 1
                l = len(english)

                # compute normalization
                for j in range(1, l+1):
                    en_word = english[j-1]
                    total_e[en_word] = 0
                    for i in range(0, m+1):
                        total_e[en_word] += t1[en_word][german[i]] * q1[(i,j,l,m)]

                # collect counts
                for j in range(1, l+1):
                    en_word = english[j-1]
                    for i in range(0, m+1):
                        g_word = german[i]
                        delta = t1[en_word][g_word] * q1[(i,j,l,m)] / total_e[en_word]
                        c1[(en_word,g_word)] += delta
                        c1[g_word] += delta
                        c1[(i,j,l,m)] += delta
                        c1[(j,l,m)] += delta
        for k in range(0, num_iter):
            for alignSent in aligned_sents:
                english = alignSent.words
                german = [None] + alignSent.mots
                m = len(german) - 1
                l = len(english)

                # compute normalization
                for j in range(1, l+1):
                    en_word = english[j-1]
                    total_e[en_word] = 0
                    for i in range(0, m+1):
                        total_e[en_word] += t[en_word][german[i]] * q[(i,j,l,m)]

                # collect counts
                for j in range(1, l+1):
                    en_word = english[j-1]
                    for i in range(0, m+1):
                        g_word = german[i]
                        delta = t[en_word][g_word] * q[(i,j,l,m)] / total_e[en_word]
                        c[(en_word,g_word)] += delta
                        c[g_word] += delta
                        c[(i,j,l,m)] += delta
                        c[(j,l,m)] += delta



            print 'calculating t and q'     
            for alignSent in aligned_sents:
                english = alignSent.words
                german = [None] + alignSent.mots
                m = len(german) - 1
                l = len(english)
                for i in range(0, m+1):
                    for j in range(1, l+1):
                        q[(i,j,l,m)] = (c[(i,j,l,m)] +  c1[(j,i,m,l)]) / (c[(j,l,m)] + c1[(i,m,l)] )
                        t[english[j-1]][german[i]] = (c[(english[j-1],german[i])] + c1[(german[i],english[j-1])])/ (c[german[i]] + c1[english[j-1]])

        return t, q


def main(aligned_sents):

    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "error.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))