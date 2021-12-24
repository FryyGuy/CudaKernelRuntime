import os
import sys
import argparse
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from Utils import get_op_config


def main():
    parser = argparse.ArgumentParser(description='CUDA Kernel Runtime')
    parser.add_argument('-ot', '-op-type', help='operation to run', required=True)
    parser.add_argument('-if', '-input-file', help='input file', required=True)
    args = vars(parser.parse_args())

    input = args["if"]
    op_type = args["ot"]

    # obtain op config based on op-type arg
    op_config = get_op_config(op_type, input)

    print(op_config)

    print("Done!")


if __name__ == "__main__":
    main()