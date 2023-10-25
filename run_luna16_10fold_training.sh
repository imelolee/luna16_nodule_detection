#!/bin/zsh

for i in 0 1 2 3 4 5 6 7 8 9
do
    python luna16_training.py -c ./config/config_train_luna16_16g.json -e ./config/environment_luna16_fold$i.json 
done