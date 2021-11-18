#
# Ruizhi Zhang
# ==============================================================================
"""Common utils."""
import os,json
import onnxruntime
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tf2.postprocess import generate_detections,transform_detections
import hparams_config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class OnnxTool:
    """This tool for onnx use within Google automl/efficientdet environment"""

    def __init__(self, config):

        self.img_size = config['output_size']
        self.crop_offset = tf.constant(config['crop_offset'])
        self.model = config['model']
        self.input_dir = config['input_dir']
        self.save_res = config['save_dir']
        self.coco_gt = config['coco_gt']

    def _preprocesser(self, file_name):
        """crawled from tf2.efficientdet_keras"""
        img = np.array(Image.open(os.path.join(self.input_dir, file_name)))
        # normalization
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = tf.cast(img, dtype=tf.float32)
        mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        img -= tf.constant(mean_rgb, shape=(1, 1, 3), dtype=tf.float32)
        img /= tf.constant(stddev_rgb, shape=(1, 1, 3), dtype=tf.float32)
        # set_scale_factors_to_output_size
        height = tf.cast(tf.shape(img)[0], tf.float32)
        width = tf.cast(tf.shape(img)[1], tf.float32)
        image_scale_y = tf.cast(self.img_size, tf.float32) / height
        image_scale_x = tf.cast(self.img_size, tf.float32) / width
        scale_factor = tf.minimum(image_scale_x, image_scale_y)
        scaled_height = tf.cast(height * scale_factor, tf.int32)
        scaled_width = tf.cast(width * scale_factor, tf.int32)
        # resize_and_crop_image
        method = tf.image.ResizeMethod.BILINEAR
        dtype = img.dtype
        scaled_image = tf.image.resize(
            img, [scaled_height, scaled_width], method=method
        )
        scaled_image = scaled_image[self.crop_offset:self.crop_offset + self.img_size,
                       self.crop_offset:self.crop_offset + self.img_size, :
                       ]
        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, self.img_size, self.img_size)
        img = tf.cast(output_image, dtype)
        # transpose
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        return img, scale_factor

    def _model_run(self, img):
        session = onnxruntime.InferenceSession(self.model)
        result = session.run([], {"input.1": img})

        return result

    def _get_model_input_names(self):
        session = onnxruntime.InferenceSession(self.model)
        input_names = session.get_inputs()
        for inp in input_names:
            print(inp.name)
        return input_names

    def _get_model_output_names(self):
        session = onnxruntime.InferenceSession(self.model)
        output_names = session.get_outputs()
        for name in output_names:
            print(name)
        return output_names

    def _post_inference(self, result, scale_factor, file_name, output_json_list, net='efficientdet-d0'):
        """crawled from tf2.postprocess for NMS"""
        result = [np.transpose(res, (0, 2, 3, 1)) for res in result]
        ### please specify which efficientdet
        config = hparams_config.get_efficientdet_config(net)

        scale_rev = 1.0 / scale_factor
        detections = generate_detections(config, result[0:5], result[5:], [scale_rev], [int(file_name[:-4])])

        det = transform_detections(detections)

        for tmp in det[0]:
            xmin = float(tmp[1])
            ymin = float(tmp[2])
            h = float(tmp[3])
            w = float(tmp[4])
            score = float(tmp[5])
            cls = int(tmp[6])
            json_dict = {'image_id': int(tmp[0]), 'category_id': cls, 'bbox': [xmin, ymin, h, w], 'score': score}
            output_json_list.append(json_dict)

        return output_json_list

    def _evaluation(self, output_json_list):
        """eval use COCO"""
        log_ofile = open(self.save_dir, 'wt')
        log_ofile.write(json.dumps(output_json_list))
        log_ofile.close()
        cocoGT = COCO(self.gt_path)
        cocoDT = cocoGT.loadRes(self.save_dir)
        cocoEval = COCOeval(cocoGT, cocoDT, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def __cosine_sim(self, data1, data2):
        """for compare results only"""
        data1, data2 = data1.flatten(), data2.flatten()
        num = data1.dot(data2.T)
        denom = np.linalg.norm(data1) * np.linalg.norm(data2)
        return num / denom

    def _compare_results_similarity(self, onnx_res, raw_res):
        """since the model outputs without NMS, there've not been the dets yet"""

        for i in range(5):

            tmp = np.fromfile(raw_res[i],dtype=np.float32)
            tmp = np.reshape(tmp, [1,int(64/(2**i)),int(64/(2**i)),810])
            tmp = np.transpose(tmp,[0,3,1,2])
            print('cls_' + str(64/(2**i)) + ':' + str(self.__cosine_sim(tmp, onnx_res[i])))

            tmp = np.fromfile(raw_res[i + 5], dtype=np.float32)
            tmp = np.reshape(tmp, [1, int(64 / (2 ** i)), int(64 / (2 ** i)), 36])
            tmp = np.transpose(tmp, [0, 3, 1, 2])
            print('bbox_' + str(64 / (2 ** i)) + ':' + str(self.__cosine_sim(tmp, onnx_res[i + 5])))

















