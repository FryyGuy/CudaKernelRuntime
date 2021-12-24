import os
import numpy as np
import json

"""
OpConfig acts as the interface for CUDA Kernels

The following fields are to be stored in JSON:
1) input file path
2) op name
3) input dimensions (to act as global dimensions for kernel launch)
"""


class GpuOpConfig(object):
    def __init__(self, input, directory, ref_op_config):
        self.ref_op_config = ref_op_config
        self.name = ""
        self.input_file_list = []
        self.directory = directory
        self.output_raw_file_path = str(self.directory) + "\\ref_outputs[0].raw"
        self.json_data = {}

        # Need to refactor to conisder multiple inputs
        file_data = np.asarray(input, dtype=np.float32)
        self.global_dims = list(file_data.shape)
        self.output_dims = list(ref_op_config.output_dims)
        self.local_dims = []

        input_raw_file = str(self.directory) + "\\inputs[0].raw"
        self.input_file_list.append(input_raw_file)
        file_data.tofile(input_raw_file, sep=" ", format="%s")

        self.output = None

    def dump_json(self):
        pass


class GrayscaleOpConfig(GpuOpConfig):
    def __init__(self, input, directory, ref_op_config):
        super().__init__(input, directory, ref_op_config)
        self.name = "rgb_to_grayscale"
        self.local_dims = [1, 1, 1]
        self.dump_json()

    def dump_json(self):
        json_file_path = str(self.directory) + "\\generated_config.json"

        self.json_data["name"] = self.name
        self.json_data["directory"] = str(self.directory)
        self.json_data["inputs"] = self.input_file_list
        self.json_data["ref_output"] = str(self.output_raw_file_path)
        self.json_data["ref_output_dims"] = self.output_dims
        self.json_data["global_dims"] = self.global_dims
        self.json_data["local_dims"] = self.local_dims
        with open(json_file_path, "w") as f:
            json.dump(self.json_data, f)



