import os
import time
import pickle
from PIL import Image
from Face import AVGFace


if __name__ == '__main__':
    # load average face
    saved_avg_face = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_avg_face.pckl')
    if os.path.isfile(saved_avg_face):
        with open(saved_avg_face, 'rb') as f:
            avg_face = pickle.load(f)

    dir_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photos')
    print('Extracting to ' + dir_save)
    for ind, face in enumerate(avg_face.faces()):
        print('Image ' + str(ind))
        Image.fromarray(face.image()).save(os.path.join(dir_save, 'face_' + str(time.time()) + '.jpeg'))

    print('Extracting the average face.')
    Image.fromarray(avg_face.image()).save(os.path.join(dir_save, 'avg_face_' + str(time.time()) + '.jpeg'))