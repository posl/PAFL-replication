for variant in srnn gru lstm; do
for method in ochiai ochiai2 dstar tarantula ample; do
python rq1_boxplot.py $variant $method
done
done