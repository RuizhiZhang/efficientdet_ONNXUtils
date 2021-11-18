#
# Ruizhi Zhang
# ==============================================================================

import json
import os
from onnx_utils import OnnxTool

if __name__ == '__main__':

    config = json.load('onnx_config.json')

    ot = OnnxTool(config)

    files_path = os.listdir(config['input_dir'])

    output_json_list = []
    for j, file_path in enumerate(files_path):
        if file_path.endswith("jpg"):
            print(file_path)
            #PREPROCESSING
            img, scale_factor = ot._preprocesser(file_path)
            #MODEL RUN
            result = ot._model_run(img)
            #INFERENCE POST
            output_json_list = ot._post_inference(result, scale_factor,
                                                  file_path, output_json_list)

    #EVALUATION
    ot._evaluation(output_json_list)
