import nltk
import A
from collections import defaultdict
import itertools

class BerkeleyAligner():
    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        x = 1

    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
        words = []
        for sent in aligned_sents:
            s =  sent.mots
            words += s
        words = set(words)
        
        t = defaultdict(int)
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
        q = defaultdict(int)
        for sent in aligned_sents: #english j foreign i
            l = len(sent.mots)
            m = len(sent.words)
            for i in range(0,l):
                for j in range(0,m):
                        q[(j,i,l,m)] += 1
        total_prob = len(q)
        for key in q.keys():
            q[key] = 1.0/total_prob
        print q
        delta = defaultdict(int)
        for s in range(0, num_iters):
            c = defaultdict(int)
            for k in range(0, len(aligned_sents)):
                german = aligned_sents[k].words
                english = aligned_sents[k].mots
                l = len(english)
                m = len(german)
                sum = 0
                for (i,j) in itertools.product(range(0,len(german)), range(0,len(english))):
                    sum += q[(j,i,l,m)]*t[(german[i], english[j])]
                delta[(k,i,j)] = q[(j,i,l,m)]*t[(german[i], english[j])]/float(sum)

            for k in range(0, len(aligned_sents)):
                german = aligned_sents[k].words
                english = aligned_sents[k].mots
                l = len(english)
                m = len(german)
                for (i,j) in itertools.product(range(0,len(german)), range(0,len(english))):
                    
                    c[(english[j], german[i])] += delta[(k,i,j)]
                    c[english[j]] += delta[(k,i,j)]
                    c[(j,i,l,m)] += delta[(k,i,j)]
                    c[(i,l,m)] += delta[(k,i,j)]

            for k in range(0, len(aligned_sents)):
                german = aligned_sents[k].words
                english = aligned_sents[k].mots
                l = len(english)
                m = len(german)
                for (i,j) in itertools.product(range(0,len(german)), range(0,len(english))):
                    t1[(german[i], english[j])] = float(c[(english[j], german[i])])/c[english[j]]
                    q1[(j,i,l,m)] = float(c[(j,i,l,m)])/c[(i,l,m)]
        print t1,q1
        return (t1,q1)

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    #A.save_model_output(aligned_sents, ba, "ba.txt")
    #avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
