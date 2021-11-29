from Reference import *
from OpConfig import *


def get_op_config(name, input, output_filename):
    if name == "rgb_to_grayscale":
        return GrayscaleOpConfig(input), GrayscaleRefConfig(input, output_filename)
