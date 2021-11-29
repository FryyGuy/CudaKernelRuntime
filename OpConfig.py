"""
OpConfig acts as the interface for CUDA Kernels

The following fields are to be stored in JSON:
1) input file path
2) op name
3) input dimensions (to act as global dimensions for kernel launch)
"""


import tensorflow as tf


class GpuOpConfig(object):
    def __init__(self, input):
        self.name = ""
        self.input = tf.cast(input, dtype=tf.float32)
        self.output = None
        self.global_dims = []
        self.local_dims = []


class GrayscaleOpConfig(GpuOpConfig):
    def __init__(self, input):
        super().__init__(input)
        self.name = "rgb_to_grayscale"
        self.global_dims = list(input.shape)

