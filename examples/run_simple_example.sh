#!/bin/bash

python create_graph.py

pygenstability run --help
pygenstability run \
    --constructor continuous_normalized \
    --min-scale -2 \
    --max-scale 0\
    --n-scale 50 \
    --n-louvain 100 \
    --n-workers 40 \
    edges.csv
#    sbm_graph.pkl

pygenstability plot_scan --help
pygenstability plot_scan results.pkl

pygenstability plot_communities --help
pygenstability plot_communities sbm_graph.gpickle results.pkl
