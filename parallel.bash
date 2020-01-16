files=( $(find results/clef/lda -name "*.prel") )
for i in $(seq 0 $(( ${#files[@]} - 1)) )
do
    path=${files[$i]}
    output=${path/results/notebooks}
    output=${output/\/pred.prel/.trec}
    echo $output
    bash run.bash trec $path > $output
done
