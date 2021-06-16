import numpy as np
import sys
from transformers import pipeline
from transformers import RobertaTokenizer, TFRobertaModel

def vec2str(x):
    vecstr = ''
    for x_i in x:
        x_i_str = '%.4f' %(x_i)
        vecstr += x_i_str + ' '
    return vecstr[0:-1] # strip off the trailing space
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ('usage: python BERT_vec_generate.py <vocab file> <outvec file>')
        sys.exit(0)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = TFRobertaModel.from_pretrained('roberta-base')
    vocabfile = sys.argv[1]
    outfile = sys.argv[2]
    nlp_features = pipeline('feature-extraction')
    with open(vocabfile) as f:
        words = f.read().splitlines()
    f = open(outfile, "w")
    # f.write(str(len(words)) + ' 768\n')
    for w in words:
        output = nlp_features(w)
        output = np.array(output)
        output = output[0]
        vec = output[1] # middle vector, first is bos, last is eos
        vecstr = vec2str(vec)
        f.write(w + '\t' + vecstr + '\n')
    f.close()