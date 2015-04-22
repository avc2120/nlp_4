import nltk
from nltk.corpus import  comtrans
from nltk.align import IBMModel1
from nltk.align import IBMModel2

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
    ibm1 = IBMModel1(aligned_sents, 10)
    return ibm1

def create_ibm2(aligned_sents):
    ibm2 = IBMModel2(aligned_sents, 10)
    return ibm2
# TODO: Initialize IBM Model 2 and return the model.
# def create_ibm2(aligned_sents):

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    errors = []
    for i in range(0,n):
        s = model.align(aligned_sents[i])
        error = s.alignment_error_rate(aligned_sents[i])
        errors.append(error)
    return sum(errors)/n
# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    f = open(file_name, 'wb')
    for i in range(0,20):
        s = model.align(aligned_sents[i])
        f.write(str(s.words) + '\n' + str(s.mots) + '\n' + str(s.alignment) + '\n\n' )

def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    # ibm2 = create_ibm2(aligned_sents)
    # save_model_output(aligned_sents, ibm2, "ibm2.txt")
    # avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    # print ('IBM Model 2')
    # print ('---------------------------')
    # print('Average AER: {0:.3f}\n'.format(avg_aer))
