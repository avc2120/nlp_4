from __future__  import division
import nltk
import A
from collections import defaultdict
import itertools
import math as math
from nltk.align import Alignment, AlignedSent
import sys


from collections import defaultdict
from nltk.align  import AlignedSent
from nltk.corpus import comtrans
from nltk.align.ibm1 import IBMModel1
class BerkeleyAligner():
    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    def align(self, align_sent):
        alignment = []
        english = align_sent.words
        german =  align_sent.mots
        l = len(english)
        m = len([None] + german)

        for j, en_word in enumerate(english):
            max_align_prob = (-sys.maxint,0)
            for i, g_word in enumerate(german):
                max_align_prob = max(max_align_prob, (self.t[en_word][g_word]*self.q[(i,j,l,m)], i))

            if max_align_prob[1] is not None:
                alignment.append((j, max_align_prob[1]))

        return AlignedSent(align_sent.words, align_sent.mots, alignment)

        
    def initialize(self, aligned_sents, num_iter, e_to_g):
        print 'getting counts'
        total_e = defaultdict(float)
        t = defaultdict(lambda: defaultdict(lambda: 0.0))
        counts = defaultdict(set)

        for alignSent in aligned_sents:
            english = alignSent.mots if not e_to_g else alignSent.words
            german = [None] + alignSent.words if not e_to_g else [None] + alignSent.mots
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
            m = len(german) 
            l = len(english)
            initial_value = 1.0 / (m )
            for i in range(0, m):
                for j in range(0, l):
                    q[(i,j,l,m)] = initial_value
        return t,q

    def train(self, aligned_sents, num_iter):
        print 'start train'
        t, q = self.initialize(aligned_sents, num_iter, True)  
        t1, q1 = self.initialize(aligned_sents, num_iter, False)  
        t.update(t1)

        c = defaultdict(float)
        c1 = defaultdict(float)
        total_e = defaultdict(float)

        print 'collecting counts'
        for k in range(0, num_iter):
            for alignSent in aligned_sents:
                english = alignSent.mots 
                german = [None] + alignSent.words
                m = len(german) 
                l = len(english)

                # compute normalization
                for j in range(0, l):
                    en_word = english[j]
                    total_e[en_word] = 0.0
                    for i in range(0, m):
                        total_e[en_word] += t[en_word][german[i]] * q1[(i,j,l,m)]

                # collect counts
                for j in range(0, l):
                    en_word = english[j]
                    for i in range(0, m):
                        g_word = german[i]
                        delta = t[en_word][g_word] * q1[(i,j,l,m)] / total_e[en_word] 
                        c1[(en_word,g_word)] += delta
                        c1[g_word] += delta
                        c1[(i,j,l,m)] += delta
                        c1[(j,l,m)] += delta

            for alignSent in aligned_sents:
                english = alignSent.words
                german = [None] + alignSent.mots
                m = len(german)
                l = len(english)

                # compute normalization
                for j in range(0, l):
                    en_word = english[j]
                    total_e[en_word] = 0.0
                    for i in range(0, m):
                        total_e[en_word] += t[en_word][german[i]] * q[(i,j,l,m)]

                # collect counts
                for j in range(0, l):
                    en_word = english[j]
                    for i in range(0, m):
                        g_word = german[i]
                        delta = t[en_word][g_word] * q[(i,j,l,m)] / total_e[en_word]
                        c[(en_word,g_word)] += delta
                        c[g_word] += delta
                        c[(i,j,l,m)] += delta
                        c[(j,l,m)] += delta


            print 'calculating t and q'     
            for alignSent in aligned_sents:
                english =  alignSent.words
                german = [None] + alignSent.mots

                english1 = alignSent.mots
                german1 = [None] + alignSent.words
                m = len(german) 
                l = len(english)

                m1 = len(german1)
                l1 = len(english1)
                for ((i,j),(i1,j1)) in zip(itertools.product(range(0, m),range(0, l)), itertools.product(range(0, m1),range(0, l1))):
                    q[(i,j,l,m)] = (c[(i,j,l,m)] +  c1[(i1,j1,l1,m1)]) / (c[(j,l,m)] + c1[(j1,l1,m1)] ) 
                    t[english[j]][german[i]] =(c[(english[j],german[i])] + c1[(english1[j1],german1[i1])])/ (c[german[i]] + c1[german1[i1]])


        return t, q


def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    #A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))