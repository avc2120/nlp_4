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
        if self.t is None or self.q is None:
            raise ValueError("The model does not train.")

        alignment = []

        l_e = len(align_sent.words)
        l_g = len(align_sent.mots)

        for j, en_word in enumerate(align_sent.words):
            
            # Initialize the maximum probability with Null token
            max_align_prob = (self.t[en_word][None]*self.q[0][j+1][l_e][l_g], None)
            for i, g_word in enumerate(align_sent.mots):
                # Find out the maximum probability
                max_align_prob = max(max_align_prob,
                    (self.t[en_word][g_word]*self.q[i+1][j+1][l_e][l_g], i))

            # If the maximum probability is not Null token,
            # then append it to the alignment. 
            if max_align_prob[1] is not None:
                alignment.append((j, max_align_prob[1]))

        return AlignedSent(align_sent.words, align_sent.mots, alignment)

        
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
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
        # q = defaultdict(float)

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

        ibm1 = IBMModel1(aligned_sents, 10)
        t = ibm1.probabilities
        t1 = ibm1.probabilities
        # Vocabulary of each language
        german_vocab = set()
        english_vocab = set()
        for alignSent in aligned_sents:
            english_vocab.update(alignSent.words)
            german_vocab.update(alignSent.mots)
        german_vocab.add(None)
        english_vocab.add(None)

        q = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float))))
        q1 = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float))))
        # Initialize the distribution of alignment probability,

        for alignSent in aligned_sents:
            english = alignSent.words
            german = [None] + alignSent.mots
            l_g = len(german) - 1
            l_e = len(english)
            initial_value = 1 / (l_g + 1)
            for (i,j) in itertools.product(range(0, l_g+1),range(1, l_e+1)):
                q[i][j][l_e][l_g] = initial_value

            english = [None] + alignSent.words
            german = alignSent.mots
            l_g = len(german) 
            l_e = len(english) -1
            initial_value = 1 / (l_g + 1)
            for (j,i) in  itertools.product(range(1, l_g+1),range(0, l_e+1)):
                q1[i][j][l_g][l_e] = initial_value
        #end
        for i in range(0, num_iter):
            count = defaultdict(lambda: defaultdict(float))
            total_g = defaultdict(float)
            count1 = defaultdict(lambda: defaultdict(float))
            total_e = defaultdict(float)

            c = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            c1 = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
            total_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
            total_align1 = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
            #start
            for alignSent in aligned_sents:
                english = alignSent.words
                german = [None] + alignSent.mots
                l_g = len(german) - 1
                l_e = len(english)

                # compute normalization
                for (j,i) in itertools.product(range(1, l_e+1),range(0, l_g+1)):
                    en_word = english[j-1]
                    total_e[en_word] = 0
                    total_e[en_word] += t[en_word][german[i]] * q[i][j][l_e][l_g]

                # collect counts
                for (j,i) in itertools.product(range(1, l_e+1),range(0, l_g+1)):
                    en_word = english[j-1]
                    g_word = german[i]
                    delta = t[en_word][g_word] * q[i][j][l_e][l_g] / total_e[en_word]
                    count[en_word][g_word] += delta
                    total_g[g_word] += delta
                    c[i][j][l_e][l_g] += delta
                    total_align[j][l_e][l_g] += delta
            #end
            for alignSent in aligned_sents:
                english = [None] + alignSent.words
                german =  alignSent.mots
                l_g = len(german) 
                l_e = len(english) -1

                # compute normalization
                for (i,j) in itertools.product(range(0, l_e+1),range(1, l_g+1)):
                    g_word = german[j-1]
                    total_g[g_word] = 0
                    total_g[g_word] += t1[g_word][english[i]] * q1[i][j][l_g][l_e]

                # collect counts
                for (i,j) in itertools.product(range(0, l_e+1),range(1, l_g+1)):
                    g_word = german[j-1]
                    en_word = english[i]
                    if total_g[g_word] != 0:
                        delta = t1[en_word][en_word] * q1[i][j][l_g][l_e] / total_g[g_word]
                        count[g_word][en_word] += delta
                        total_g[en_word] += delta
                        c1[i][j][l_g][l_e] += delta
                        total_align1[j][l_g][l_e] += delta
                    else:
                        print 'divide 0'
            #-----
            # estimate probabilities
            t = defaultdict(lambda: defaultdict(lambda: 0.0))
            q = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))

            # Smoothing the counts for alignments
            # for alignSent in aligned_sents:
            #     english = alignSent.words
            #     german = [None] + alignSent.mots
            #     l_g = len(german) - 1
            #     l_e = len(english)

            #     laplace = 1.0
            #     for i in range(0, l_g+1):
            #         for j in range(1, l_e+1):
            #             value = c[i][j][l_e][l_g]
            #             if 0 < value < laplace:
            #                 laplace = value

            #     laplace *= 0.5 
            #     for i in range(0, l_g+1):
            #         for j in range(1, l_e+1):
            #             c[i][j][l_e][l_g] += laplace

            #     initial_value = laplace * l_e
            #     for j in range(1, l_e+1):
            #         total_align[j][l_e][l_g] += initial_value
            
            # Estimate the new lexical translation probabilities
            for g,e in itertools.product(german_vocab, english_vocab):
                t[e][g] = (count[e][g] + count1[g][e])/ (total_g[g] + total_e[e])

            # Estimate the new alignment probabilities
            for alignSent in aligned_sents:
                english = alignSent.words
                german = [None] + alignSent.mots
                l_g = len(german) - 1
                l_e = len(english)
                for i,j in itertools.product( range(0, l_g+1),range(1, l_e+1)):
                    q[i][j][l_e][l_g] = (c[i][j][l_e][l_g])/ (total_align[j][l_e][l_g])

        return t, q

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    #A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
