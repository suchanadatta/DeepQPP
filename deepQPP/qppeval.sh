#!/bin/bash

if [ $# -le 6 ] 
then
    echo "Usage: " $0 " <following arguments in the order>";
    echo "1. Path of the AP file.";
    echo "2. Path of the interaction matrix (separate file for each qid).";
    echo "3. Training batch size.";
    echo "4. No. of epochs.";
    echo "5. No. of cross validation folds.";
    echo "6. Type of evaluation : pair/point.";	
    echo "7. Want to save predicted values? -- yes/no.";
    exit 1;
fi

apPath=`readlink -f $1`
qPairJudge=$(dirname $apPath)
qPairJudge=$qPairJudge"/judge.pairs"

interMatrixPath=`readlink -f $2`
interMatrixPath=$interMatrixPath"/"
batchSize=$3
epochs=$4
cvFolds=$5
evalType=$6
result=$7

echo "Using AP file at: "$apPath
echo "Store query pair judgement file at : "$qPairJudge

# making the .properties file
cat > qppeval.properties << EOL
[Section]

apPath=$apPath

qPairJudge=$qPairJudge

interMatrixPath=$interMatrixPath

batchSize=$batchSize

epochs=$epochs

cvFolds=$cvFolds

evalType=$evalType

result=$result

EOL
# .properties file made

python3 ../query_pair_judgement.py qppeval.properties

echo "\nUsing interaction matrix at: "$interMatrixPath

python3 deep_qpp_driver.py qppeval.properties


