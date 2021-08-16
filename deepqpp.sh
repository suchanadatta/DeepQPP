#!/bin/bash

# create interaction matrices

cd ./InteractionMatrix/

echo "#######################################################################"
echo "################# Generate Interaction Matrices #######################"
echo "#######################################################################"

sh interaction.sh ../data/query /store/index/trec678/ ../InteractionMatrix/resources/smart-stopwords 3 100 ../data/ /store/causalIR/drmm/data/trec678.vec.model.txt content ../data/interaction_matrix/

# store per query AP values
# provide proper initial retrieved data file name

./resources/trec_eval-master/trec_eval -q -m all_trec ./data/qrel ./data/LMDirichlet1000.0-D10-content.res | awk '{if ($1=="map" && $2!="all") print $2"\t"$3}' > ./data/per_query_ap

# run deepqpp learning module

cd ../deepQPP/

echo "\n#####################################################################"
echo "##################### Running DeepQPP Module ##########################"
echo "#######################################################################"

sh qppeval.sh ./data/per_query_ap ./data/interaction_matrix 10 1 5 point yes

echo "\n#####################################################################"
echo "############################## DONE ###################################"
echo "#######################################################################"

