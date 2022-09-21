for variant in srnn gru lstm; do
for method in ochiai ochiai2 dstar tarantula ample; do
python kn_dataset_relation.py $variant $method
done
done