#!/bin/bash

set -e # Abort if one of the commands fail

python Utest_ex_vs_rand.py
python count_significant.py