BIN_SIZE=20
K=10
N=100

#N is the number of queries
#Data (histogram) for each query is 10x20 dimensional

for n in $(seq 1 1 $N)
do
  cp /dev/null data/$n.txt
  for k in $(seq 1 1 $k)
  do
    for i in $(seq 1 1 $BIN_SIZE)
    do
      printf "$(($RANDOM % 100)) " >> data/$n.txt
    done
    printf "\n" >> data/$n.txt
  done
done

# Generate the id-id label random file

cp /dev/null data/pairs.txt
e=$(expr $N - 1)

for i in $(seq 1 1 $e)
do
  s=$(expr $i + 1)
  for j in $(seq $s 1 $N)
  do
    printf "$i\t$j\t$(($RANDOM % 2))\n" >> data/pairs.txt
  done
done

