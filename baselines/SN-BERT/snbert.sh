#!/bin/bash

if [ $# -le 6 ]
then
    echo "Usage: " $0 " <following arguments in the order>";
    echo "1. Path of the query file.";
    echo "2. Path of the query BERT vectors (separate file for each qid).";
    echo "3. Path of the AP file.";
    echo "4. Training batch size.";
    echo "5. No. of epochs.";
    echo "6. No. of cross validation folds.";
    echo "7. Type of evaluation : pair/point.";
    echo "8. Want to save predicted values? -- yes/no.";
    exit 1;
fi

# generate query term BERT vectors
queryPath=`readlink -f $1`
bertVecPath=`readlink -f $2`
bertVecPath=$bertVecPath"/"

# generate the pairwise judgement from AP values
apPath=`readlink -f $3`
qPairJudge=$(dirname $apPath)
qPairJudge=$qPairJudge"/judge.pairs"

batchSize=$4
epochs=$5
cvFolds=$6
evalType=$7
result=$8

echo "Using query path at: "$queryPath
echo "Query term BERT vectors will be saved at: "$bertVecPath
echo "Using AP file at: "$apPath
echo "Store query pair judgement file at : "$qPairJudge

# making the .properties file
cat > snbert.properties << EOL
[Section]

queryPath=$queryPath

bertVecPath=$bertVecPath

apPath=$apPath

qPairJudge=$qPairJudge

batchSize=$batchSize

epochs=$epochs

cvFolds=$cvFolds

evalType=$evalType

result=$result

EOL
# .properties file made

python3 ../../query_pair_judgement.py snbert.properties

echo "\nUsing query BERT vectors at: "$bertVecPath

python3 sn_bert_driver.py snbert.properties


