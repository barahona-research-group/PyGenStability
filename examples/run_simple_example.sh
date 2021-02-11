<<<<<<< Updated upstream
#!/bin/bash

python create_graph.py

pygenstability run --help
pygenstability run \
    --constructor continuous_normalized \
    --min-time -2 \
    --max-time 0\
    --n-time 50 \
    --n-louvain 100 \
    --n-workers 40 \
    edges.csv
#    sbm_graph.pkl

pygenstability plot_scan --help
=======
#!/bin/zsh

python create_graph.py
pygenstability run --n-time 100 sbm_graph.pkl
>>>>>>> Stashed changes
pygenstability plot_scan results.pkl

pygenstability plot_communities --help
pygenstability plot_communities sbm_graph.gpickle results.pkl
