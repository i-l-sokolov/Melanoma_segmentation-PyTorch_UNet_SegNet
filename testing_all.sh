#!/bin/bash

epoch=${1:-20}
batch=${2:-25}


MODELS=({2..3})
LOSSES=({0..14})

if [ -f "/results/results.csv" ]; then
  rm "results/results.csv"
fi

cd lib

for model in "${MODELS[@]}"
do
  for loss in "${LOSSES[@]}"
  do
    python assembling.py --model $model --loss $loss --epoch $epoch --batch_size $batch
    wait
  done
done

python plotting_results.py
