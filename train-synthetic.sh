#!/bin/bash

./synthetic.d --norm=1 --lambda=1 > synthetic/l1-1.csv
./synthetic.d --norm=1 --lambda=2 > synthetic/l1-2.csv
./synthetic.d --norm=1 --lambda=4 > synthetic/l1-4.csv
./synthetic.d --norm=1 --lambda=8 > synthetic/l1-8.csv
./synthetic.d --norm=1 --lambda=16 > synthetic/l1-16.csv
./synthetic.d --norm=1 --lambda=inf > synthetic/l1-inf.csv

./synthetic.d --norm=2 --lambda=1 > synthetic/l2-1.csv
./synthetic.d --norm=2 --lambda=2 > synthetic/l2-2.csv
./synthetic.d --norm=2 --lambda=4 > synthetic/l2-4.csv
./synthetic.d --norm=2 --lambda=8 > synthetic/l2-8.csv
./synthetic.d --norm=2 --lambda=16 > synthetic/l2-16.csv
./synthetic.d --norm=2 --lambda=inf > synthetic/l2-inf.csv

./synthetic.d --norm=inf --lambda=1 > synthetic/linf-1.csv
./synthetic.d --norm=inf --lambda=2 > synthetic/linf-2.csv
./synthetic.d --norm=inf --lambda=4 > synthetic/linf-4.csv
./synthetic.d --norm=inf --lambda=8 > synthetic/linf-8.csv
./synthetic.d --norm=inf --lambda=16 > synthetic/linf-16.csv
./synthetic.d --norm=inf --lambda=inf > synthetic/linf-inf.csv

./synthetic.d --norm=1 --batchnorm --lambda=1 > synthetic/l1-1-batchnorm.csv
./synthetic.d --norm=2 --batchnorm --lambda=1 > synthetic/l2-1-batchnorm.csv
./synthetic.d --norm=inf --batchnorm --lambda=1 > synthetic/linf-1-batchnorm.csv
