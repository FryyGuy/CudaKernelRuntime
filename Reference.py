import numpy as np
import tensorflow as tf
import time


class ReferenceConfig(object):
    def __init__(self, input, directory):
        self.input = tf.convert_to_tensor(input)
        self.output = tf.zeros(input.shape)
        self.output_dims = list()
        self.output_directory = directory
        self.output_filename = str(self.output_directory) + "\\ref_output.jpeg"
        self.execution_time = float(0)

    def execute(self):
        pass

    def __str__(self):
        config_str = ""
        config_str += "input          = " + str(self.input) + "\n"
        config_str += "output         = " + str(self.output) + "\n"
        config_str += "execution_time = " + str(self.execution_time) + "\n"
        return config_str


class GrayscaleRefConfig(ReferenceConfig):
    def __init__(self, input, directory):
        super().__init__(input, directory)
        self.execute()

    def execute(self):
        self.output = tf.constant(self.input)
        t_start = time.time()
        self.output = tf.image.rgb_to_grayscale(self.output)
        self.output_dims = self.output.shape
        t_end = time.time()
        self.execution_time = float(t_end - t_start)
        write_output = tf.image.encode_jpeg(self.output)
        tf.io.write_file(filename=self.output_filename, contents=write_output)

        ref_output = np.asarray(self.output)
        output_raw_file = str(self.output_directory) + "\\ref_outputs[0].raw"
        ref_output.tofile(output_raw_file, sep=" ", format="%s")
