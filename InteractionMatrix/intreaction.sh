#!/bin/bash

if [ $# -le 6 ] 
then
    echo "Usage: " $0 " <following arguments in the order>";
    echo "1. Query file path.";
    echo "2. Path of the lucene index."
    echo "3. Stopwords file path."
    echo "4. Initial retrieved documents.";
    echo "5. Word vector file path.";
    echo "6. Name of the field used for searching.";
    echo "7. Interaction matrix path.";
    exit 1;
fi

queryPath=`readlink -f $1`
indexPath=`readlink -f $2`
stopFilePath=`readlink -f $3`
retFilePath=`readlink -f $4`
wordVecPath=`readlink -f $5`
searchField=$6
interMatrixPath=`readlink -f $7`
interMatrixPath=$interMatrixPath"/"

echo "Using query file at: "$queryPath
echo "Using index at : "$indexPath
echo "Using stop file at : "$stopFilePath
echo "Using initial retrieved/preranked file at :"$retFilePath
echo "Using word2vec file at : "$wordVecPath 
echo "Store interaction matrices at :"$interMatrixPath

# making the .properties file
cat > interaction.properties << EOL

queryPath=$queryPath

indexPath=$indexPath

stopFilePath=$stopFilePath

retFilePath=$retFilePath

wordVecPath=$wordVecPath

searchField=$searchField

interMatrixPath=$interMatrixPath

EOL
# .properties file made

java -Xmx3g -cp $CLASSPATH:dist/InteractionMatrix.jar interactionmatrix.GenerateHistogramPrerankFile
