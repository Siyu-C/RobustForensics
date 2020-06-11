#!/usr/bin/env sh
cd image_based
sh train.sh configs/b0/config.yaml $1 $2 $3
sh train.sh configs/b1/config.yaml $1 $2 $3
sh train.sh configs/b1/config_alldata.yaml $1 $2 $3
sh train.sh configs/b1/config_long_alldata.yaml $1 $2 $3
sh train.sh configs/b3/config.yaml $1 $2 $3
sh train.sh configs/r34/config.yaml $1 $2 $3
sh train.sh configs/xcep/config.yaml $1 $2 $3
cd ../video_based
sh train.sh configs/config.yaml $1 $2 $3
sh train.sh configs/config_noaug.yaml $1 $2 $3
sh train.sh configs/config_16x8.yaml $1 $2 $3
sh train.sh configs/config_alldata.yaml $1 $2 $3
cd ..
