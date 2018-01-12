declare -a datasets=(2DMOT2015)
declare -a type=(train test)
OUTPUT_DIRECTORY=$1

for dataset in "${datasets[@]}"
do
    for t in "${type[@]}"
    do
        mkdir -p ../build/$OUTPUT_DIRECTORY/$dataset/$t/
        ls data/$dataset/$t/ > data/$dataset/$t/sequences.lst
        sed -i '/sequences.lst/d' data/$dataset/$t/sequences.lst
        while read sequence;
        do
            echo $dataset,$t,$sequence
            ls data/$dataset/$t/$sequence/img1/ > data/$dataset/$t/$sequence/img1/pos.lst
            sed -i '/pos.lst/d' data/$dataset/$t/$sequence/img1/pos.lst
            ./detector_mars data/$dataset/$t/$sequence/ > $OUTPUT_DIRECTORY/$dataset/$t/$sequence.txt
            rm data/$dataset/$t/$sequence/img1/pos.lst
        done <./data/$dataset/train/sequences.lst
        #rm data/$dataset/$t/sequences.lst
    done
done
