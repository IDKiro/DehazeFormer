#!/usr/bin/env bash

python test.py --model dehazeformer-t --dataset RESIDE-IN --exp indoor
python test.py --model dehazeformer-s --dataset RESIDE-IN --exp indoor
python test.py --model dehazeformer-b --dataset RESIDE-IN --exp indoor
python test.py --model dehazeformer-w --dataset RESIDE-IN --exp indoor
python test.py --model dehazeformer-d --dataset RESIDE-IN --exp indoor
python test.py --model dehazeformer-m --dataset RESIDE-IN --exp indoor
python test.py --model dehazeformer-l --dataset RESIDE-IN --exp indoor

python test.py --model dehazeformer-t --dataset RESIDE-OUT --exp outdoor
python test.py --model dehazeformer-s --dataset RESIDE-OUT --exp outdoor
python test.py --model dehazeformer-b --dataset RESIDE-OUT --exp outdoor
python test.py --model dehazeformer-m --dataset RESIDE-OUT --exp outdoor

python test.py --model dehazeformer-t --dataset RESIDE-6K --exp reside6k
python test.py --model dehazeformer-s --dataset RESIDE-6K --exp reside6k
python test.py --model dehazeformer-b --dataset RESIDE-6K --exp reside6k
python test.py --model dehazeformer-m --dataset RESIDE-6K --exp reside6k

python test.py --model dehazeformer-t --dataset RSHaze --exp rshaze
python test.py --model dehazeformer-s --dataset RSHaze --exp rshaze
python test.py --model dehazeformer-b --dataset RSHaze --exp rshaze
python test.py --model dehazeformer-m --dataset RSHaze --exp rshaze
