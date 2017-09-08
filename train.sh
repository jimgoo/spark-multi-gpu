#!/bin/bash
set -e

MIN_GPUS=1
MAX_GPUS=8
EPOCHS=10

for N_GPUS in $(seq $MIN_GPUS $MAX_GPUS)
do
    python keras_rnn.py train --n_gpus $N_GPUS --epochs $EPOCHS --backend tensorflow
    KERAS_BACKEND=mxnet python keras_rnn.py train --n_gpus $N_GPUS --epochs $EPOCHS --backend mxnet
    python pytorch_rnn.py train --n_gpus $N_GPUS --epochs $EPOCHS
done
