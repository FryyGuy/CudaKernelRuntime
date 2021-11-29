import os
import sys
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from Utils import get_op_config
from OpConfig import *
from Reference import *

def main():
    parser = argparse.ArgumentParser(description='CUDA Kernel Runtime')
    parser.add_argument('-ot', '-op-type', help='operation to run', required=True)
    parser.add_argument('-if', '-input-file', help='input file', required=True)
    parser.add_argument('-of', '-output-file', help='output file', required=True)
    args = vars(parser.parse_args())

    img = args["if"]
    op_type = args["ot"]
    output_filename = args["of"]

    img_data = tf.io.read_file(img)
    orig_pic = tf.io.decode_jpeg(img_data)

    # obtain op config based on op-type arg
    op_config, ref_op_config = get_op_config(op_type, orig_pic, output_filename)

    print(ref_op_config)

    print("Done!")

if __name__ == "__main__":
    main()