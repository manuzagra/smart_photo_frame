# import the necessary packages
import os
import time
import pickle
from picamera.array import PiRGBArray
from picamera import PiCamera
from SmartFrameGUI import SmartFrameGUI
from ImageFace import ImageFace
from Face import AVGFace


class SmartFrameApp:
    def __init__(self):

        # Camera
        self._camera = PiCamera()
        self._camera.resolution = '720p'
        self._camera.framerate = 5

        # average face
        self._saved_avg_face = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_avg_face.pckl')
        if os.path.isfile(self._saved_avg_face):
            with open(self._saved_avg_face, 'rb') as f:
                self._avg_face = pickle.load(f)
        else:
            self._avg_face = AVGFace([], height=800, width=480)

        # GUI
        self._gui = SmartFrameGUI(height=480, width=800)
        self._gui.display_image(self._avg_face.image(), 1)


    def run(self):

        raw_capture = PiRGBArray(self._camera)

        time_between_detections = 10
        n_previews = 10
        photo_counter = 0
        state = 'detecting'

        for frame in self._camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            image = ImageFace(frame.array, height=800, width=480)

            print(state)
            if state == 'detecting':
                if image.faces():
                    state = 'preview'

            elif state == 'preview':
                if image.faces():
                    self._gui.display_image(image.faces()[0].image(), 1)
                photo_counter = photo_counter + 1

                if photo_counter >= n_previews:
                    state = 'calculating'

            elif state == 'calculating':
                if image.faces():
                    # calculate the average
                    self._avg_face = AVGFace(self._avg_face.faces() + image.faces())
                    self._gui.display_image(self._avg_face.image(), 1)
                    photo_counter = 0
                    state = 'waiting'
                    with open(self._saved_avg_face, 'wb') as f:
                        pickle.dump(self._avg_face, f)

            elif state == 'waiting':
                time.sleep(time_between_detections)
                state = 'detecting'

            self._gui.refresh()

            # clear the stream in preparation for the next frame
            raw_capture.truncate(0)


class SmartFrameAppNoPreview:
    def __init__(self):

        # Camera
        self._camera = PiCamera()
        self._camera.resolution = '720p'
        self._camera.framerate = 5

        # average face
        self._saved_avg_face = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_avg_face.pckl')
        if os.path.isfile(self._saved_avg_face):
            with open(self._saved_avg_face, 'rb') as f:
                self._avg_face = pickle.load(f)
        else:
            self._avg_face = AVGFace([], height=800, width=480)

        # GUI
        self._gui = SmartFrameGUI(height=480, width=800)
        self._gui.display_image(self._avg_face.image(), 1)


    def run(self):

        raw_capture = PiRGBArray(self._camera)

        time_between_detections = 10
        n_previews = 10
        photo_counter = 0
        state = 'detecting'

        for frame in self._camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            image = ImageFace(frame.array, height=800, width=480)

            print(state)
            if state == 'detecting':
                if image.faces():
                    state = 'preview'

            elif state == 'preview':
#                 if image.faces():
#                     self._gui.display_image(image.faces()[0].image(), 1)
                photo_counter = photo_counter + 1

                if photo_counter >= n_previews:
                    state = 'show'

            elif state == 'show':
                self._gui.display_image(image.image(), 1)
                time.sleep(2)
                state = 'calculating'

            elif state == 'calculating':
                if image.faces():
                    # calculate the average
                    self._avg_face = AVGFace(self._avg_face.faces() + image.faces())
                    self._gui.display_image(self._avg_face.image(), 1)
                    photo_counter = 0
                    state = 'waiting'
                    with open(self._saved_avg_face, 'wb') as f:
                        pickle.dump(self._avg_face, f)

            elif state == 'waiting':
                time.sleep(time_between_detections)
                state = 'detecting'

            self._gui.refresh()

            # clear the stream in preparation for the next frame
            raw_capture.truncate(0)


class SmartFrameAppLimited:
    def __init__(self, faces_lenght=float("inf")):

        # Camera
        self._camera = PiCamera()
        self._camera.resolution = '720p'
        self._camera.framerate = 5

        # average face
        self._faces_lenght = faces_lenght
        self._saved_avg_face = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_avg_face.pckl')
        if os.path.isfile(self._saved_avg_face):
            with open(self._saved_avg_face, 'rb') as f:
                self._avg_face = pickle.load(f)
        else:
            self._avg_face = AVGFace([], height=800, width=480)

        # GUI
        self._gui = SmartFrameGUI(height=480, width=800)
        self._gui.display_image(self._avg_face.image(), 1)


    def run(self):

        raw_capture = PiRGBArray(self._camera)

        time_between_detections = 10
        n_previews = 10
        photo_counter = 0
        state = 'detecting'

        for frame in self._camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            image = ImageFace(frame.array, height=800, width=480)

            print(state)
            if state == 'detecting':
                if image.faces():
                    state = 'preview'

            elif state == 'preview':
                if image.faces():
                    self._gui.display_image(image.faces()[0].image(), 1)
                photo_counter = photo_counter + 1

                if photo_counter >= n_previews:
                    state = 'show'

            elif state == 'show':
                self._gui.display_image(image.image(), 1)
                time.sleep(2)
                state = 'calculating'

            elif state == 'calculating':
                if image.faces():
                    # calculate the
                    faces = self._avg_face.faces() + image.faces()
                    if len(faces) > self._faces_lenght:
                        del faces[0:(len(faces)-self._faces_lenght)]
                    self._avg_face = AVGFace(faces)
                    self._gui.display_image(self._avg_face.image(), 1)
                    photo_counter = 0
                    state = 'waiting'
                    with open(self._saved_avg_face, 'wb') as f:
                        pickle.dump(self._avg_face, f)

            elif state == 'waiting':
                time.sleep(time_between_detections)
                state = 'detecting'

            self._gui.refresh()

            # clear the stream in preparation for the next frame
            raw_capture.truncate(0)


if __name__ == "__main__":

    # if you want to have a limit in the number of faces just add it as an argument
    #gui = SmartFrameAppLimited(50)
    gui = SmartFrameApp()
    gui.run()
