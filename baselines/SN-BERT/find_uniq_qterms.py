import sys, re
import xml.etree.ElementTree as ET
from nltk.stem import PorterStemmer


if len(sys.argv) < 3:
    print('Needs 2 arguments - \n'
          '1. TREC query file\n'
          '2. Output file with unique query terms.\n')
    exit(0)

arg_query_file = sys.argv[1]
arg_res_file = sys.argv[2]

initial_query_list = []

def extract_qterms_uniq(query_file):
    stemmer = PorterStemmer()
    rootElement = ET.parse(query_file).getroot()
    for subElement in rootElement:
        query = re.sub('[^a-zA-Z0-9\n\.]', ' ', subElement[1].text)
        query_terms = query.split()
        print(query_terms)
        for term in query_terms:
            term_stem = stemmer.stem(term.lower().strip())
            print(term_stem)
            if term_stem not in initial_query_list:
                initial_query_list.append(term_stem)
                print(initial_query_list)

extract_qterms_uniq(arg_query_file)
f = open(arg_res_file, "w")
for term in initial_query_list:
    f.write(term + '\n')
f.close()
