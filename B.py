import nltk
import A
from collections import defaultdict
import itertools
import math as math
from nltk.align import Alignment, AlignedSent
import sys
class BerkeleyAligner():
    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        alignments = []
        english = [None] +  align_sent.mots
        german = align_sent.words
        l = len(english)
        m = len(german)
        for i in range(0,len(german)):
            maximum = -sys.maxint
            argmax = None
            for j in range(1,len(english)):
                prod = math.log(self.q[(j,i,l,m)],2) + math.log(self.t[(german[i], english[j])],2)
                if prod > maximum:
                    maximum = prod
                    argmax = (i,j)
            alignments.append(argmax)
            print argmax
        sent = AlignedSent(german, english, Alignment(alignments))
        return sent

        
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
        words = []
        for sent in aligned_sents:
            s =  sent.mots
            words += s
        words = set(words)
        
        t = defaultdict(float)
        q = defaultdict(float)
        delta = defaultdict(float)
        c = defaultdict(float)
        t1 = defaultdict(float)
        q1 = defaultdict(float)

        for word in words:
            t[word] = []
            possibles = []
            for sent in aligned_sents:
                s = sent.mots
                if word in s:
                    possibles += sent.words
            possibles = set(possibles)
            length = len(possibles)
            for possible in possibles:
                t[(possible, word)] = 1/float(length)
        
        for sent in aligned_sents: #english j foreign i
            english =[None] +  sent.mots
            german = sent.words
            l = len(english)
            m = len(german)
            for (i,j) in itertools.product(range(0,len(german)), range(1,len(english))):
                q[(j,i,l,m)] += 1.0
        total_prob = len(q)
        for key in q.keys():
            q[key] = 1.0/total_prob


        for s in range(0, num_iters):
            for k in range(0, len(aligned_sents)):
                german =  aligned_sents[k].words
                english = [None] +aligned_sents[k].mots
                l = len(english)
                m = len(german)
                sum = 0.0
                for (i,j) in itertools.product(range(0,len(german)), range(1,len(english))):
                    sum += float(q[(j,i,l,m)])*float(t[(german[i], english[j])])
                for (i,j) in itertools.product(range(0,len(german)), range(1,len(english))):
                    delta[(k,i,j)] = float(q[(j,i,l,m)])*float(t[(german[i], english[j])])/float(sum)
        for s in range(0, num_iters):
            for k in range(0, len(aligned_sents)):
                german = aligned_sents[k].words
                english = [None] + aligned_sents[k].mots
                l = len(english)
                m = len(german)
                for (i,j) in itertools.product(range(0,len(german)), range(1,len(english))):
                    c[(english[j], german[i])] += float(delta[(k,i,j)])
                    c[english[j]] += float(delta[(k,i,j)])
                    c[(j,i,l,m)] += float(delta[(k,i,j)])
                    c[(i,l,m)] += float(delta[(k,i,j)])

        for s in range(0, num_iters):
            for k in range(0, len(aligned_sents)):
                german = aligned_sents[k].words
                english = [None] +aligned_sents[k].mots
                l = len(english)
                m = len(german)
                for (i,j) in itertools.product(range(0,len(german)), range(1,len(english))):
                    if float(c[(english[j], german[i])])/float(c[english[j]]) == 0:
                        t[(german[i], english[j])] = -1000
                    else:
                        t[(german[i], english[j])] = float(c[(english[j], german[i])])/float(c[english[j]])
                    if float(c[(j,i,l,m)])/float(c[(i,l,m)]) == 0:
                        q[(j,i,l,m)] = -1000
                    else:
                        q[(j,i,l,m)] = float(c[(j,i,l,m)])/float(c[(i,l,m)])
        return (t,q)

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    #A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
