# it makes the pairwise judgement for query pairs
# inputs = qid \t ap file
# output = binary class label - 1/0 (1 -> q1>q2; 0 -> q1<q2)
# output format - q1 \t q2 \t class_label

import configparser
import sys

class QueryPairJudgement:

    def __init__(self, prop_file):
        self.prop_file = prop_file
        self.config = configparser.ConfigParser()
        self.config.read(self.prop_file)
        self.qid_ap_dict = {}

    def make_qid_ap_dict(self):
        apFile = self.config.get('Section', 'apPath')
        fp = open(apFile)
        for line in fp.readlines():
            parts = line.rstrip().split('\t')
            self.qid_ap_dict[parts[0]] = parts[1]

    def write_query_pair_judge(self):
        judgePath = open(self.config.get('Section', 'qPairJudge'), 'w')
        qid_list = self.qid_ap_dict.keys()
        for id in qid_list:
            curr_qid = id
            for entry in qid_list:
                if entry > curr_qid:
                    if self.qid_ap_dict[curr_qid] > self.qid_ap_dict[entry]:
                        judgePath.writelines(curr_qid + '\t' + entry + '\t' + '1\n')
                    else:
                        judgePath.writelines(curr_qid + '\t' + entry + '\t' + '0\n')
        print('======= Complete writing query pair judgements =======')

qpj = QueryPairJudgement(sys.argv[1])
qpj.make_qid_ap_dict()
qpj.write_query_pair_judge()
