import cv2

from pathlib import Path

from Reference import *
from OpConfig import *


# Need to refactor to consider multiple inputs
def get_op_config(name, img_path):
    input = cv2.imread(img_path)
    directory = Path(img_path).parent.absolute()
    if name == "rgb_to_grayscale":
        ref_op_config = GrayscaleRefConfig(input, directory)
        return GrayscaleOpConfig(input, directory, ref_op_config)
