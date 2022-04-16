"""
file: driver.py
description: CSCI 630, Lab 3, Wikipedia Language Classification
language: Python
author: Abhishek Shah, as5553
"""

import sys
from decisionTree import *
from adaBoost import *


def main():
    """
    Main Function
    Training: python3 driver.py train <examples> <hypothesisOut> <learning-type(dt or ada)>
    Testing: python3 driver.py predict <hypothesis> <file> <testing-type(dt or ada)>
    """
    if sys.argv[1] == 'train':
        if sys.argv[4] == 'ada':
            adaDataCollection(sys.argv[2], sys.argv[3])
        else:
            dtDataCollection(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'predict':
        if sys.argv[4] == 'ada':
            adaPredict(sys.argv[2], sys.argv[3])
        else:
            dtPredict(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
