import os
import dlib
import cv2
import Face


class ImageFace:
    def __init__(self, image, detector=cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cv_haarcascades_trained_models', 'haarcascade_frontalface_default.xml')), height=800, width=480):
        """
        This class will take an image, extract and process all the faces
        :param img: input image
        :param detector: the face detector to use
        :param height: height of the saved faces
        :param width: width of the saved faces
        """

        # save the image
        self._image = image

        # the shape
        self._shape = (height, width)

        # get all the rectangles containing faces detected in the image
        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        self._detections = detector.detectMultiScale(gray, 1.3, 5)

        # save each face in the image
        self._faces = [Face.Face(self._image, dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h), height=height, width=width) for (x, y, w, h) in self._detections]

    def faces(self):
        return self._faces

    def image(self):
        return self._image

    def detections(self):
        return self._detections
