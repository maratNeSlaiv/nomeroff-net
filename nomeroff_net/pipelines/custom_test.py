from typing import Any, Dict, Optional, List, Union

import numpy as np

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
# from  import needed_function
# from parseq-main import OCR_parseq
# from strhub.Untitled_1 import OCR_parseq
from nomeroff_net.pipelines.parseq_main_file import OCR_parseq




class AnyNumberPlateDetectionAndReading(Pipeline, CompositePipeline):
    """
    Number Plate Detection And Reading Class
    """

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
        """
        init NumberPlateDetectionAndReading Class
        Args:
            image_loader (): image_loader
            path_to_model (): path_to_model
            mtl_model_path (): mtl_model_path
            refiner_model_path (): refiner_model_path
            path_to_classification_model (): path_to_classification_model
            presets (): presets
            off_number_plate_classification (): off_number_plate_classification
            classification_options (): classification_options
            default_label (): default_label
            default_lines_count (): default_lines_count
            number_plate_localization_class (): number_plate_localization_class
            number_plate_localization_detector (): number_plate_localization_detector

        """
        self.default_label = default_label
        self.default_lines_count = default_lines_count
        self.number_plate_localization = number_plate_localization_class(
            "number_plate_localization",
            image_loader=None,
            path_to_model=path_to_model,
            detector=number_plate_localization_detector
        )
        self.number_plate_key_points_detection = NumberPlateKeyPointsDetection(
            "number_plate_key_points_detection",
            image_loader=None,
            mtl_model_path=mtl_model_path,
            refiner_model_path=refiner_model_path)
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
        self.number_plate_text_reading = NumberPlateTextReading(
            "number_plate_text_reading",
            image_loader=None,
            presets=presets,
            option_detector_width=option_detector_width,
            option_detector_height=option_detector_height,
            default_label=default_label,
            default_lines_count=default_lines_count,
            off_number_plate_classification=off_number_plate_classification,
        )
        self.pipelines = [
            self.number_plate_localization,
            self.number_plate_key_points_detection,
            self.number_plate_text_reading,
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


    def forward(self, inputs: Any, **forward_parameters: Dict):
        images_bboxs, images = unzip(self.number_plate_localization(inputs, **forward_parameters))
        images_points, images_mline_boxes = unzip(self.number_plate_key_points_detection(unzip([images, images_bboxs]),
                                                                                         **forward_parameters))
        
        zones, image_ids = crop_number_plate_zones_from_images(images, images_points)

        cv2.imwrite('trash_of_mina_1.jpg' , zones[0])

        if self.number_plate_classification is None or not len(zones):
            region_ids = [-1 for _ in zones]
            region_names = [self.default_label for _ in zones]
            count_lines = [self.default_lines_count for _ in zones]
            confidences = [-1 for _ in zones]
            predicted = [-1 for _ in zones]
            preprocessed_np = [None for _ in zones]
        else:
            (region_ids, region_names, count_lines,
             confidences, predicted, preprocessed_np) = unzip(self.number_plate_classification(zones,
                                                                                               **forward_parameters))
            
            if count_lines[0] == 2:
                
                # SQUARE NUMBER PLATE RECOGNITION

                image = zones[0]
                image = Image.fromarray(image)

                # Верх и низ (может понадобится потом)
                width, height = image.size
                midpoint = height // 2
                upper_image = image.crop((0, 0, width, midpoint))
                lower_image = image.crop((0, midpoint, width, height))

                                
                # Левая и правая картинка 
                width, height = image.size
                width_part1 = width // 3
                width_part2 = width - width_part1  
                box1 = (0,0, width_part1, height)
                box2 = (width_part1, 0, width, height)
                left_half = image.crop(box1)
                right_half = image.crop(box2)

                # Лева и право делим еще пополам каждую
                width, height = left_half.size
                midpoint = height // 2
                upper_left = left_half.crop((0, 0, width, midpoint))
                lower_left = left_half.crop((0, midpoint, width, height))

                width, height = right_half.size
                midpoint = height // 2
                upper_right = right_half.crop((0, 0, width, midpoint))
                lower_right = right_half.crop((0, midpoint, width, height))
      
                # upper_left.save('trash_1_upper_left_1.jpg')
                # upper_right.save('trash_1_upper_right_1.jpg')
                # lower_left.save('trash_1_lower_left_1.jpg')
                # lower_right.save('trash_1_lower_right_1.jpg')
                # upper_image.save('trash_1_upper_1.jpg')
                # lower_image.save('trash_1_lower_1.jpg')

                # Проверка левого верхнего угла 
                important_check = OCR_parseq(upper_left)

                # функция чтобы оставить только символы и цифры
                import re
                pattern = re.compile(r'[^a-zA-Z0-9]')
                important_check = pattern.sub('', important_check)
                    
                # Проверка на две цифры в левом верхнем углу (кыргызский номер - регион)
                pattern = re.compile(r'^\d{2}$')
                if pattern.match(important_check):
                    texts = [[
                        important_check + 
                        # " <- region " +
                        str(OCR_parseq(upper_right)) +
                        # + '  |||||   ' + 
                        str(OCR_parseq(lower_right)) 
                        # + "-> Not my output ->"
                        ,
                    ]]
                else:
                    # Только одна Буква или цифра (абхазский номер)
                    pattern = re.compile(r'^[a-zA-Z0-9]$')
                    if pattern.match(important_check):
                        texts = [[
                            # '<- not my output <- ' +
                        str(OCR_parseq(upper_image)) +  
                        # + '  |||||   ' + 
                        str(OCR_parseq(lower_image)),
                        # + "-> Not my output ->",
                        ]]

                    # Пока тут только армянские номера
                    else:
                        texts = [[
                            # '<- not my output <- ' +
                            str(OCR_parseq(upper_right)) + 
                            # '  |||||   ' + 
                            str(OCR_parseq(lower_image))
                            #   + "-> Not my output ->",
                        ]]


                print('THE WORK OF THE OCR_PARSEQ MODEL')
                
                return [images, images_bboxs,
                      images_points, zones,
                      region_ids, region_names,
                      count_lines, confidences, texts]


                # count = 0

                # Leave of a quadratic
            
                


            
            zones = convert_multiline_images_to_one_line(
                image_ids,
                images,
                zones,
                images_mline_boxes,
                images_bboxs,
                count_lines,
                region_names,
            )
            
            (region_ids, region_names, count_lines,
            confidences, predicted, preprocessed_np) = unzip(self.number_plate_classification(zones,
                                                                                               **forward_parameters))
        print('''ITS THE WORK OF RECTANGULAR MODEL
              
              
              ''')
        return self.rectangular_recognition(region_ids, region_names,
                                           count_lines, confidences,
                                           zones, image_ids,
                                           images_bboxs, images,
                                           images_points, preprocessed_np, **forward_parameters)
    
    def quadratic_recognition(self, image: Any):
        # Put the OCR_parseq recognition code here, when all tests are passed
        return
    

    def rectangular_recognition(self, region_ids, region_names,
                               count_lines, confidences,
                               zones, image_ids,
                               images_bboxs, images,
                               images_points, preprocessed_np, **forward_parameters):

        number_plate_text_reading_res = unzip(
            self.number_plate_text_reading(unzip([zones,
                                                  region_names,
                                                  count_lines, preprocessed_np]), **forward_parameters))

        if len(number_plate_text_reading_res):
            texts, _ = number_plate_text_reading_res
        else:
            texts = []
        (region_ids, region_names, count_lines, confidences, texts, zones) = \
            group_by_image_ids(image_ids, (region_ids, region_names, count_lines, confidences, texts, zones))
        return [images, images_bboxs,
                      images_points, zones,
                      region_ids, region_names,
                      count_lines, confidences, texts]

    @empty_method
    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        return inputs


class CustomPipline(AnyNumberPlateDetectionAndReading):

    def custom_transform_image(self, image):
        image = image[..., ::-1]
        return image
    
    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:       
        images = [self.custom_transform_image(item) for item in inputs]
        return images

def get_text_and_region(pipline: CustomPipline, frame: np.array) -> tuple[str, str] | None:
    (images, images_bboxs, 
    images_points, images_zones, region_ids, 
    region_names, count_lines, 
    confidences, texts) = pipline([frame])

    print('''WE GOT THERE MAAAN!
          
          ''')

    if texts[0] and region_names:
        return texts[0][0] # , region_names[0][0], images_zones[0][0]
    return


if __name__ == '__main__':
    pipline = CustomPipline('number_plate_detection_and_reading_runtime', image_loader='cv2')
    frame = cv2.imread('/Users/maratorozaliev/Desktop/image_5_.jpg')
    res = get_text_and_region(pipline, frame)
    print(res)
    # print(res[0], res[1])


    # Уберем слэши тэги и тирешки если вдруг неправильно прочла 
    # res = str(res[0]) + str(res[1])
    import re
    pattern = re.compile(r'[^a-zA-Z0-9]')
    res = pattern.sub('', res)
    # cv2.imwrite('/Users/adilet/nomeroff-net/media/2_res.jpg', res[2])
    print(res)

    # print(res)