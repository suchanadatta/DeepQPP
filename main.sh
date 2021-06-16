#!/bin/bash

cd ./InteractionMatrix/

echo "#######################################################################"
echo "################# Generate Interaction Matrices #######################"
echo "#######################################################################"

sh intreaction.sh /home/suchana/NetBeansProjects/NeuralModelQpp/query.xml /store/index/trec678/ /home/suchana/smart-stopwords /store/causalIR/model-aware-qpp/lm-dir_res/lm_dir_top100 /store/causalIR/drmm/data/trec678.vec.model.txt content ../data/

cd ../deepQPP/

echo "\n#######################################################################"
echo "##################### Running DeepQPP Module ##########################"
echo "#######################################################################"

sh qppeval.sh /home/suchana/PycharmProjects/DeepQPP/data/per_query_ap_401-450 /home/suchana/PycharmProjects/DeepQPP/data/interaction_matrix 1 1 5 pair yes

echo "============== DONE ==============="

