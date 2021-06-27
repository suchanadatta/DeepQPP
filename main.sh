#!/bin/bash

cd ./InteractionMatrix/

echo "#######################################################################"
echo "################# Generate Interaction Matrices #######################"
echo "#######################################################################"

sh intreaction.sh ./data/query /store/index/trec678/ ./InteractionMatrix/resources/smart-stopwords 3 100 ./data/ /store/causalIR/drmm/data/trec678.vec.model.txt content ./data/interaction_matrix/

cd ../deepQPP/

echo "\n#####################################################################"
echo "##################### Running DeepQPP Module ##########################"
echo "#######################################################################"

sh qppeval.sh /home/suchana/PycharmProjects/DeepQPP/data/per_query_ap_401-450 /home/suchana/PycharmProjects/DeepQPP/data/interaction_matrix 10 1 5 point yes

echo "============== DONE ==============="

