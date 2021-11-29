import tensorflow as tf
import time


class ReferenceConfig(object):
    def __init__(self, input, output_filename):
        self.input = input
        self.output = tf.zeros(input.shape)
        self.output_filename = output_filename
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
    def __init__(self, input, output_filename):
        super().__init__(input=input, output_filename=output_filename)
        self.execute()

    def execute(self):
        self.output = tf.constant(self.input)
        t_start = time.time()
        self.output = tf.image.rgb_to_grayscale(self.output)
        t_end = time.time()
        self.execution_time = float(t_end - t_start)
        write_output = tf.image.encode_jpeg(self.output)
        tf.io.write_file(filename=self.output_filename, contents=write_output)
