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

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        # alignments = []
        # english = ['STOP'] +  align_sent.mots
        # german = align_sent.words
        # l = len(english)
        # m = len(german)
        # for i in range(0,m):
        #     maximum = -sys.maxint
        #     argmax = None
        #     for j in range(0,l):
        #         if self.q[(j,i,l,m)] == 0 or self.t[(german[i], english[j])] == 0:
        #             prod = -1000
        #         else:
        #             prod = math.log(self.q[(j,i,l,m)],2) + math.log(self.t[(german[i], english[j])],2)
        #         if prod > maximum:
        #             maximum = prod
        #             argmax = (i,j)
        #     alignments.append(argmax)
        # sent = AlignedSent(german, english, Alignment(alignments))
        # return sent

        alignment = []

        l_e = len(align_sent.words)
        l_g = len(align_sent.mots)

        for j, en_word in enumerate(align_sent.words):
            
            # Initialize the maximum probability with Null token
            max_align_prob = (self.t[en_word][None]*self.q[(0,j+1,l_e,l_g)], None)
            for i, fr_word in enumerate(align_sent.mots):
                # Find out the maximum probability
                max_align_prob = max(max_align_prob,
                    (self.t[en_word][fr_word]*self.q[(i+1,j+1,l_e,l_g)], i))

            # If the maximum probability is not Null token,
            # then append it to the alignment. 
            if max_align_prob[1] is not None:
                alignment.append((j, max_align_prob[1]))

        return AlignedSent(align_sent.words, align_sent.mots, alignment)

        
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train2(self, aligned_sents, num_iter):
        print 'training 2'
        ibm1 = IBMModel1(aligned_sents, 10)
        t_eg = ibm1.probabilities

        # Vocabulary of each language
        german_vocab = set()
        english_vocab = set()
        for alignSent in aligned_sents:
            english_vocab.update(alignSent.mots)
            german_vocab.update(alignSent.words)
        german_vocab.add(None)

        q = defaultdict(float)

        # Initialize the distribution of alignment probability,
        # a(i|j,l_e, l_g) = 1/(l_g + 1)

        for alignSent in aligned_sents:
            english = alignSent.mots
            german = [None] + alignSent.words
            l_g = len(german) - 1
            l_e = len(english)
            initial_value = 1 / (l_g + 1)
            for i in range(0, l_g+1):
                for j in range(1, l_e+1):
                    q[(i,j,l_e,l_g)] = initial_value

        print 'collecting counts'
        for i in range(0, num_iter):
            count_eg = defaultdict(float)
            total_g = defaultdict(float)

            c = defaultdict(float)
            total_align = defaultdict(float)
            total_e = defaultdict(float)

            for alignSent in aligned_sents:
                english = alignSent.mots
                german = [None] + alignSent.words
                l_g = len(german) - 1
                l_e = len(english)

                # compute normalization
                for j in range(1, l_e+1):
                    en_word = english[j-1]
                    total_e[en_word] = 0
                    for i in range(0, l_g+1):
                        total_e[en_word] += t_eg[en_word][german[i]] * q[(i,j,l_e,l_g)]

                # collect counts
                for j in range(1, l_e+1):
                    en_word = english[j-1]
                    for i in range(0, l_g+1):
                        fr_word = german[i]
                        delta = t_eg[en_word][fr_word] * q[(i,j,l_e,l_g)] / total_e[en_word]
                        count_eg[(en_word,fr_word)] += delta
                        total_g[fr_word] += delta
                        c[(i,j,l_e,l_g)] += delta
                        total_align[(j,l_e,l_g)] += delta
        return count_eg, total_g, c, total_align
    def train(self, aligned_sents, num_iter):
        # words = []
        # for sent in aligned_sents:
        #     s =  sent.mots
        #     words += s
        # words = set(words)
        
        # t = defaultdict(float)
        # q = defaultdict(float)
        # delta = defaultdict(float)
        # c = defaultdict(float)
        # t1 = defaultdict(float)
        # q1 = defaultdict(float)

        # for word in words:
        #     t[word] = []
        #     possibles = []
        #     for sent in aligned_sents:
        #         s = sent.mots
        #         if word in s:
        #             possibles += sent.words
        #     possibles = set(possibles)
        #     length = len(possibles)
        #     for possible in possibles:
        #         t[(possible, word)] = 1/float(length)
        
        # for sent in aligned_sents: #english j foreign i
        #     english =['STOP'] +  sent.mots
        #     german = sent.words
        #     l = len(english)
        #     m = len(german)
        #     for (i,j) in itertools.product(range(0,m), range(0,l)):
        #         q[(j,i,l,m)] += 1.0
        # total_prob = len(q)
        # for key in q.keys():
        #     q[key] = 1.0/total_prob


        # for s in range(0, num_iters):
        #     for (k, sent) in enumerate(aligned_sents):
        #         german =  sent.words
        #         english = ['STOP'] + sent.mots
        #         l = len(english)
        #         m = len(german)
        #         sum = 0.0
        #         for (i, g) in enumerate(german):
        #             for (j,e) in enumerate(english):
        #                 sum += q[(j,i,l,m)]*t[(g, e)]
        #             for (j,e) in enumerate(english):
        #                 delta[(k,i,j)] = float(q[(j,i,l,m)])*t[(g, e)]/float(sum)
        
        # for s in range(0, num_iters):
        #     for k in range(0, len(aligned_sents)):
        #         german = aligned_sents[k].words
        #         english = ['STOP'] + aligned_sents[k].mots
        #         l = len(english)
        #         m = len(german)
        #         for (i,j) in itertools.product(range(0,m), range(1,l)):
        #             c[(english[j], german[i])] += float(delta[(k,i,j)])
        #             c[english[j]] += float(delta[(k,i,j)])
        #             c[(j,i,l,m)] += float(delta[(k,i,j)])
        #             c[(i,l,m)] += float(delta[(k,i,j)])

        # for s in range(0, num_iters):
        #     for k in range(0, len(aligned_sents)):
        #         german = aligned_sents[k].words
        #         english = ['STOP'] + aligned_sents[k].mots
        #         l = len(english)
        #         m = len(german)
        #         for (i,j) in itertools.product(range(0,m), range(1,l)):
        #             print float(c[english[j]])
        #             t[(german[i], english[j])] = float(c[(english[j], german[i])])/float(c[english[j]])
        #             q[(j,i,l,m)] = c[(j,i,l,m)]/float(c[(i,l,m)])
        # print t,q
        # return (t,q)
        print 'start train1s'
        ibm1 = IBMModel1(aligned_sents, 10)
        t_eg = ibm1.probabilities

        # Vocabulary of each language
        german_vocab = set()
        english_vocab = set()
        for alignSent in aligned_sents:
            english_vocab.update(alignSent.words)
            german_vocab.update(alignSent.mots)
        german_vocab.add(None)

        q = defaultdict(float)

        # Initialize the distribution of alignment probability,
        # a(i|j,l_e, l_g) = 1/(l_g + 1)

        for alignSent in aligned_sents:
            english = alignSent.words
            german = [None] + alignSent.mots
            l_g = len(german) - 1
            l_e = len(english)
            initial_value = 1 / (l_g + 1)
            for i in range(0, l_g+1):
                for j in range(1, l_e+1):
                    q[(i,j,l_e,l_g)] = initial_value
        print 'collecting train 1'
        count_eg1, total_g1, c1, total_align1 = self.train2(aligned_sents, num_iter) 
        for i in range(0, num_iter):
            count_eg = defaultdict(float)
            total_g = defaultdict(float)

            c = defaultdict(float)
            total_align = defaultdict(float)

            total_e = defaultdict(float)

            for alignSent in aligned_sents:
                english = alignSent.words
                german = [None] + alignSent.mots
                l_g = len(german) - 1
                l_e = len(english)

                # compute normalization
                for j in range(1, l_e+1):
                    en_word = english[j-1]
                    total_e[en_word] = 0
                    for i in range(0, l_g+1):
                        total_e[en_word] += t_eg[en_word][german[i]] * q[(i,j,l_e,l_g)]

                # collect counts
                for j in range(1, l_e+1):
                    en_word = english[j-1]
                    for i in range(0, l_g+1):
                        fr_word = german[i]
                        delta = t_eg[en_word][fr_word] * q[(i,j,l_e,l_g)] / total_e[en_word]
                        count_eg[(en_word,fr_word)] += delta
                        total_g[fr_word] += delta
                        c[(i,j,l_e,l_g)] += delta
                        total_align[(j,l_e,l_g)] += delta
            
            # estimate probabilities

            t_eg = defaultdict(lambda: defaultdict(lambda: 0.0))
            q = defaultdict(float)

            # Estimate the new lexical translation probabilities
            print 'calculating t and q'
            for f in german_vocab:
                for e in english_vocab:
                    t_eg[e][f] = (count_eg[(e,f)] + count_eg1[(f,e)])/ (total_g[f] + total_g1[e])

            # Estimate the new alignment probabilities
            for alignSent in aligned_sents:
                english = alignSent.words
                german = [None] + alignSent.mots
                l_g = len(german) - 1
                l_e = len(english)
                for i in range(0, l_g+1):
                    for j in range(1, l_e+1):
                        q[(i,j,l_e,l_g)] = (c[(i,j,l_e,l_g)] +  c1[(j,i,l_g,l_e)]) / (total_align[(j,l_e,l_g)] + total_align1[(i,l_g,l_e)] )

        return t_eg, q


def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    #A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))