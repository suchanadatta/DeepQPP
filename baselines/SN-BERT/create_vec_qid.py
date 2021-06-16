import sys, re
import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer
import numpy as np
from transformers import pipeline
from transformers import RobertaTokenizer, TFRobertaModel

query_dict = {}

def make_qterm_dict(query_file):
    stemmer = PorterStemmer()
    max_qterm = 0
    rootElement = ET.parse(query_file).getroot()
    for subElement in rootElement:
        query = re.sub('[^a-zA-Z0-9\n\.]', ' ', subElement[1].text)
        query_terms = query.split()
        if len(query_terms) > max_qterm:
            max_qterm = len(query_terms)
        # print(query_terms)
        qterm_list = []
        for term in query_terms:
            qterm_list.append(stemmer.stem(term.lower().strip()))
            # print(term_stem)
        query_dict[subElement[0].text.strip()] = qterm_list
    return max_qterm

def vec2str(x):
    vecstr = ''
    for x_i in x:
        x_i_str = '%.4f' %(x_i)
        vecstr += x_i_str + ' '
    return vecstr[0:-1] # strip off the trailing space

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ('usage: python create-vec_qid.py <trec query file> <outvec file path>')
        sys.exit(0)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = TFRobertaModel.from_pretrained('roberta-base')
    queryfile = sys.argv[1]
    max_qterm = make_qterm_dict(queryfile)
    print(query_dict)
    print(max_qterm)
    outfilepath = sys.argv[2]
    nlp_features = pipeline('feature-extraction')
    for qid in query_dict:
        qterms = query_dict.get(qid)
        if len(qterms) == max_qterm:
            with open(outfilepath + qid + '.bert', 'w') as f:
                for t in qterms:
                    output = nlp_features(t)
                    output = np.array(output)
                    output = output[0]
                    vec = output[1]  # middle vector, first is bos, last is eos
                    vecstr = vec2str(vec)
                    f.write(t + ' ' + vecstr + '\n')
            f.close()
        else:
            diff = max_qterm - len(qterms)
            with open(outfilepath + qid + '.bert', 'w') as f:
                for t in qterms:
                    output = nlp_features(t)
                    output = np.array(output)
                    output = output[0]
                    vec = output[1]  # middle vector, first is bos, last is eos
                    vecstr = vec2str(vec)
                    f.write(t + ' ' + vecstr + '\n')
                while diff > 0:
                    f.write('x' + ' ')
                    for i in range(1, 769):
                        f.write('0.0' + ' ')
                    f.write('\n')
                    diff = diff - 1
            f.close()




