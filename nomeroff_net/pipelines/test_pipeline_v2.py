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
from PIL import Image
import numpy as np
import torch
from some import OCR_parseq


class custom_pipeline_v2(Pipeline, CompositePipeline):
    
    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 path_to_model: str = "latest",
                 mtl_model_path: str = "latest",
                 refiner_model_path: str = "latest",
                 path_to_classification_model: str = "latest",
                 presets: Dict = None,
                 off_number_plate_classification: bool = False,
                 classification_options: List = None,
                 default_label: str = "eu_ua_2015",
                 default_lines_count: int = 1,
                 number_plate_localization_class: Pipeline = DefaultNumberPlateLocalization,
                 number_plate_localization_detector=None,
                 **kwargs):
        self.default_label = default_label
        self.default_lines_count = default_lines_count
        self.number_plate_localization = number_plate_localization_class(
            "number_plate_localization",
            image_loader=None,
            path_to_model=path_to_model,
            detector=number_plate_localization_detector
        )
        self.number_plate_classification = None
        option_detector_width = 0
        option_detector_height = 0

        if not off_number_plate_classification:
            self.number_plate_classification = NumberPlateClassification(
                "number_plate_classification",
                image_loader=None,
                path_to_model=path_to_classification_model,
                options=classification_options)
            option_detector_width = self.number_plate_classification.detector.width
            option_detector_height = self.number_plate_classification.detector.height

        self.pipelines = [
            self.number_plate_localization,
        ]

        if self.number_plate_classification is not None:
            self.pipelines.append(self.number_plate_classification)
        Pipeline.__init__(self, task, image_loader, **kwargs)
        CompositePipeline.__init__(self, self.pipelines)

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        images = [self.image_loader.load(item) for item in inputs]
        return images


    def quadratic_detection(self, inputs: Any, **forward_parameters: Dict):
        images_bboxs, images = unzip(self.number_plate_localization(inputs, **forward_parameters))
        
        # Чекер это координаты номера 
        # Используется чтобы игнорировать крафт модель
        # Надо адоптировать чтобы читабельней было

        checker = ([[
            [ np.float64(images_bboxs[0][0][0]), np.float64(images_bboxs[0][0][3]) ], 
            [ np.float64(images_bboxs[0][0][0]), np.float64(images_bboxs[0][0][1]) ], 
            [ np.float64(images_bboxs[0][0][2]), np.float64(images_bboxs[0][0][1]) ], 
            [ np.float64(images_bboxs[0][0][2]), np.float64(images_bboxs[0][0][3]) ]
        ]],)

        zones, image_ids = crop_number_plate_zones_from_images(images, checker)

        #Расчет на то что предиктим только квадратные номера
        (region_ids, region_names, count_lines,
        confidences, predicted, preprocessed_np) = unzip(self.number_plate_classification(zones,
                                                                                               **forward_parameters))

            
        if(count_lines[0] == 2):
            image = zones[0]
            height, width = image.shape[:2]
            midpoint = height // 2
            upper_half = Image.fromarray( image[:midpoint, :] )
            lower_half = Image.fromarray( image[midpoint:, :] ) 
            
            # Нужно вынести за функцию чтобы один раз загружать.
            parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
            region_and_numbers = OCR_parseq(parseq, upper_half)
            bottom_characters = OCR_parseq(parseq, lower_half)
            return region_and_numbers, bottom_characters
            
            # cv2.imwrite('trash_of_mine_night_upper_half.jpg', upper_half)
            # cv2.imwrite('trash_of_mine_night_lower_half.jpg', lower_half)
        
        else:
            return "Используйте модель только для квадратных номеров с двумя линиями", "Пожалуйста"


if __name__ == '__main__':
    
    frame = cv2.imread('/Users/maratorozaliev/Desktop/image_5_.jpg')
    pipline = custom_pipeline_v2(image_loader='cv2')
    upper_symbols, lower_symbols = pipline.quadratic_detection(frame)
    
    print('''Upper symbols:
          
          ''', upper_symbols,
          '''
          Lower symbols:
          
          ''', lower_symbols)