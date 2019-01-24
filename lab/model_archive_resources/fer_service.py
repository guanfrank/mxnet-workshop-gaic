# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np
from mxnet_model_service import MXNetModelService
from mxnet_utils import image, ndarray
from skimage import transform
import mxnet as mx
import cv2 as cv
import logging

# One time initialization of Haar Cascade Classifier to extract and crop out face
face_detector = cv.CascadeClassifier('haarcascade_frontalface.xml')
# Classifier parameter specifying how much the image size is reduced at each image scale
scale_factor = 1.3
# Classifier parameter how many neighbors each candidate rectangle should have to retain it
min_neighbors = 5

def crop_face(image):
    """Attempts to identify a face in the input image.

    Parameters
    ----------
    image : array representing a BGR image

    Returns
    -------
    array
        The cropped face, transformed to grayscale. If no face found returns None

    """
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_roi_list = face_detector.detectMultiScale(gray_image, scale_factor, min_neighbors)
    
    if (len(face_roi_list) > 0):
        (x,y,w,h) = face_roi_list[0]
        return gray_image[y:y+h,x:x+w]
    else:
        return None

class FERService(MXNetModelService):
    """
    Defines custom pre and post processing for the Facial Emotion Recognition model
    """

    def preprocess(self, request):
        """
        Pre-process requests by attempting to extract face image, and transforing to fit the model's input
        Parameters
        ----------
        data : list of input images
            Raw inputs from request.
        Returns
        -------
        list of NDArray
            Processed images in the model's expected input shape
        """
        img_list = []

        for idx, data in enumerate(request):
            param_name = self.signature['inputs'][idx]['data_name']
            input_shape = self.signature['inputs'][idx]['data_shape']
            img = data.get(param_name)
            if img is None:
                img = data.get("body")

            if img is None:
                img = data.get("data")

            if img is None or len(img) == 0:
                self.error = "Empty image input"
                return None

            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]

            try:
                img_arr = image.read(img)
            except Exception as e:
                logging.warn(e, exc_info=True)
                self.error = "Corrupted image input"
                return None

            face = crop_face(img_arr.asnumpy())
            if (face is not None):
                face = transform.resize(face, (h,w))            
            # If no face identified - use the entire input image
            else:
                face = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
            img_arr = np.resize(face, input_shape)
            img_list.append(mx.nd.array(img_arr))
        return img_list

    def postprocess(self, data):
        response = []
        for d in data:
            inference_result = d.softmax().asnumpy()
        for idx, label in enumerate(self.labels):
            response.append({label: float(inference_result[0][idx])})
        return [response]


_service = FERService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
