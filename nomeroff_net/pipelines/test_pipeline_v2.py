"""number plate detection and reading pipeline

Examples:
    >>> from nomeroff_net import pipeline
    >>> from nomeroff_net.tools import unzip
    >>> number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", image_loader="opencv")
    >>> results = number_plate_detection_and_reading(['./data/examples/oneline_images/example1.jpeg', './data/examples/oneline_images/example2.jpeg'])
    >>> (images, images_bboxs, images_points, images_zones, region_ids,region_names, count_lines, confidences, texts) = unzip(results)
    >>> print(texts)
    (['AC4921CB'], ['RP70012', 'JJF509'])
"""
from typing import Any, Dict, Optional, List, Union

import numpy as np
import sys
sys.path.append('/Users/maratorozaliev/Desktop/nomeroff-net')

from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline, CompositePipeline, empty_method
from nomeroff_net.pipelines.number_plate_localization import NumberPlateLocalization as DefaultNumberPlateLocalization
from nomeroff_net.pipelines.number_plate_key_points_detection import NumberPlateKeyPointsDetection
from nomeroff_net.pipelines.number_plate_text_reading import NumberPlateTextReading
from nomeroff_net.pipelines.number_plate_classification import NumberPlateClassification
from nomeroff_net.tools.image_processing import crop_number_plate_zones_from_images, group_by_image_ids
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_multiline_extractors.multiline_np_extractor \
    import convert_multiline_images_to_one_line, convert_multiline_to_one_line
import cv2
from nomeroff_net.pipelines.test_pipeline import CustomPipline, get_text_and_region_one


if __name__ == '__main__':
    frame = cv2.imread('/Users/maratorozaliev/Desktop/2024-05-15_18.25.39.jpg')
    pipline = CustomPipline('number_plate_reading_runtime', image_loader='cv2')
    
    res = get_text_and_region_one(pipline, frame)
    cv2.imwrite('/Users/maratorozaliev/Desktop/test_of_1.jpg', res[2])
    print(res[0], res[1])
