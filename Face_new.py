import os
import dlib
import cv2
import numpy as np
import math
from multiprocessing.pool import ThreadPool


def similarity_transform(input_points, output_points):
    """
    The function takes a pair of source points and a destination points and returns the transform between them.
    :param inPoints: pair of input points
    :param outPoints: pair of output points
    :return:
    """

    # cv needs at least 3 points to calculate the transform so we add to each pair of points a third one, creating a rectangle triangle
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    in_pts = np.copy(input_points).tolist()
    out_pts = np.copy(output_points).tolist()

    xin = c60 * (in_pts[0][0] - in_pts[1][0]) - s60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = s60 * (in_pts[0][0] - in_pts[1][0]) + c60 * (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]
    in_pts.append([np.int(xin), np.int(yin)])

    xout = c60 * (out_pts[0][0] - out_pts[1][0]) - s60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = s60 * (out_pts[0][0] - out_pts[1][0]) + c60 * (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]
    out_pts.append([np.int(xout), np.int(yout)])

    # calculate the transform between the two sets of points
    tform = cv2.estimateRigidTransform(np.array([in_pts]), np.array([out_pts]), fullAffine=False)

    return tform


def calc_boundary_points(width=0, height=0):
    return np.array([(0, 0), (width/2, 0), (width-1, 0), (width-1, height/2), (width-1, height-1), (width/2, height-1), (0, height-1), (0, height/2)], dtype='float32')


# Apply affine transform calculated using src_tri and dst_tri to img and output an image of size.
def apply_affine_transform(src, src_tri, dst_tri, size):
    # Given a pair of triangles, find the affine transform.
    transform = cv2.getAffineTransform(src_tri, dst_tri)

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, transform, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warp_triangle(src_img, dst_img, src_tri, dst_tri):
    # Find bounding rectangle for each triangle
    src_rect = cv2.boundingRect(src_tri.astype(np.float32))
    dst_rect = cv2.boundingRect(dst_tri.astype(np.float32))

    # Offset points by left top corner of the respective rectangles
    src_tri_rect = [((p[0] - src_rect[0]), (p[1] - src_rect[1])) for p in src_tri]
    dst_tri_rect = [((p[0] - dst_rect[0]), (p[1] - dst_rect[1])) for p in dst_tri]

    # Get mask by filling triangle
    mask = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    cutted_src_img = src_img[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]

    size = (dst_rect[2], dst_rect[3])

    warpped_cutted_src_img = apply_affine_transform(cutted_src_img, np.float32(src_tri_rect), np.float32(dst_tri_rect), size)

    warpped_cutted_src_img = warpped_cutted_src_img * mask

    # Copy triangular region of the rectangular patch to the output image
    dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] = dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] * (
                (1.0, 1.0, 1.0) - mask)

    dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] = dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] + warpped_cutted_src_img


class Face:
    def __init__(self, image, detection, predictor=dlib.shape_predictor(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dlib_trained_face_shape_predictor_model', 'shape_predictor_68_face_landmarks.dat')), height=800, width=480):
        """
        This class takes an image, a face detection, a predictor and the shape of the final image. It transform the input face into the desired shape and it rotates the image to put the eyes in the correct place.
        :param image:
        :param detection:
        :param predictor:
        :param height:
        :param width:
        """

        # the final position of the eyes
        # final_eyecorners = [(np.int(0.35 * width), np.int(height / 3)), (np.int(0.65 * width), np.int(height / 3))]
        final_eyecorners = [(np.int(0.3 * width), np.int(height / 3)), (np.int(0.7 * width), np.int(height / 3))]

        # calculate the landmarks of the faces detected in the image and convert them into np.array
        lmk = predictor(image, detection)
        lmk = np.array([(p.x, p.y) for p in lmk.parts()])

        # corners of the eye in input image
        source_eyecorner = [lmk.astype(int).tolist()[36], lmk.astype(int).tolist()[45]]

        # compute similarity transform
        tform = similarity_transform(source_eyecorner, final_eyecorners)

        # apply transform
        self._image = cv2.warpAffine(image.astype(np.float32), tform, (width, height))

        # apply similarity transform on landmarks
        lmk = np.reshape(np.array(lmk).astype(float), (68, 1, 2))
        lmk = cv2.transform(lmk, tform)
        self._landmarks = np.float32(np.reshape(lmk, (68, 2)))

        # the boundary points around the image
        self._boundary = calc_boundary_points(height=self.shape()[0], width=self.shape()[1])

    def shape(self):
        """
        :return: the shape of the image [0] - height, [1] - width
        """
        return self._image.shape[0:2]

    def image(self):
        return self._image.astype(np.uint8)

    def landmarks(self):
        return self._landmarks

    def boundary(self):
        return self._boundary

    def points_of_interest(self):
        return np.concatenate((self._landmarks, self._boundary))

    def warp_to_AVGFace(self, avg_face):
        # intialize the image
        img = np.zeros(shape=(self.shape()[0], self.shape()[1], 3), dtype='float32')

        # in every triangle we warp the image to fit the average triangle
        for triangle in avg_face.delaunay_triangles():
            tri_in = np.array([self.points_of_interest()[triangle[0]], self.points_of_interest()[triangle[1]], self.points_of_interest()[triangle[2]]], dtype='float32')
            tri_out = np.array([avg_face.points_of_interest()[triangle[0]], avg_face.points_of_interest()[triangle[1]], avg_face.points_of_interest()[triangle[2]]], dtype='float32')

            warp_triangle(self._image, img, tri_in, tri_out)

        return img


class AVGFace:
    def __init__(self, faces=[], height=800, width=480):
        """
        It suppose all the images have the same size than this one
        :param faces:
        :param height:
        :param width:
        """

        self._image = np.zeros(shape=(height, width, 3), dtype='float32')
        self._landmarks = np.zeros(shape=(68, 2), dtype='float32')
        self._boundary = calc_boundary_points(height=self.shape()[0], width=self.shape()[1])

        self._delaunay_triangles = []

        self._faces = faces
        self._n_faces = len(faces)

        if self._faces:
            # calculate the average landmaks
            for face in faces:
                self._landmarks = self._landmarks + face.landmarks()
            self._landmarks = self._landmarks / self._n_faces

            # calculate the delaunay triangles
            self.update_delaunay_triangles()

            # warp all the faces to match our average landmarks and calculate the average
            pool = ThreadPool(5)
            warped_faces = pool.map(lambda x: x.warp_to_AVGFace(self), faces)
            for face in warped_faces:
                self._image = self._image + face
            self._image = self._image / self._n_faces

    def update_delaunay_triangles(self):
        # Points to work with
        points = self.points_of_interest().tolist()

        # Create subdiv
        subdiv = cv2.Subdiv2D((0, 0, self.shape()[1], self.shape()[0]))
        subdiv.insert(points)

        # Find the indices of triangles in the points array
        delaunay = []

        for t in subdiv.getTriangleList():
            # if all the points are inside the image
            if self.point_in([t[0], t[1]]) and self.point_in([t[2], t[3]]) and self.point_in([t[4], t[5]]):
                # find the index of those points in the landmarks + boundry array
                ind = (points.index([t[0], t[1]]), points.index([t[2], t[3]]), points.index([t[4], t[5]]))
                delaunay.append(ind)

        self._delaunay_triangles = delaunay

    def shape(self):
        """
        :return: the shape of the image
        """
        return self._image.shape[0:2]

    def image(self):
        return self._image.astype('uint8')

    def faces(self):
        return self._faces

    def landmarks(self):
        return self._landmarks

    def boundary(self):
        return self._boundary

    def points_of_interest(self):
        return np.concatenate((self._landmarks, self._boundary))

    def delaunay_triangles(self):
        return self._delaunay_triangles

    def point_in(self, point):
        if point[0] < 0:
            return False
        elif point[1] < 0:
            return False
        elif point[0] > self.shape()[1]:
            return False
        elif point[1] > self.shape()[0]:
            return False
        return True


class AVGFaceLimited:
    def __init__(self, faces=[], faces_lenght=float("inf"), height=800, width=480):
        """
        It suppose all the images have the same size than this one
        :param faces:
        :param height:
        :param width:
        """

        self._image = np.zeros(shape=(height, width, 3), dtype='float32')
        self._landmarks = np.zeros(shape=(68, 2), dtype='float32')
        self._boundary = calc_boundary_points(height=self.shape()[0], width=self.shape()[1])

        self._delaunay_triangles = []

        if len(faces) > faces_lenght:
            raise ValueError('Too many faces. The maximum number of face is {}'.format(faces_lenght))

        self._faces = faces
        self._n_faces = len(faces)

        if self._faces:
            # calculate the average landmaks
            for face in faces:
                self._landmarks = self._landmarks + face.landmarks()
            self._landmarks = self._landmarks / self._n_faces

            # calculate the delaunay triangles
            self.update_delaunay_triangles()

            # warp all the faces to match our average landmarks and calculate the average
            pool = ThreadPool(5)
            warped_faces = pool.map(lambda x: x.warp_to_AVGFace(self), faces)
            for face in warped_faces:
                self._image = self._image + face
            self._image = self._image / self._n_faces

    def update_delaunay_triangles(self):
        # Points to work with
        points = self.points_of_interest().tolist()

        # Create subdiv
        subdiv = cv2.Subdiv2D((0, 0, self.shape()[1], self.shape()[0]))
        subdiv.insert(points)

        # Find the indices of triangles in the points array
        delaunay = []

        for t in subdiv.getTriangleList():
            # if all the points are inside the image
            if self.point_in([t[0], t[1]]) and self.point_in([t[2], t[3]]) and self.point_in([t[4], t[5]]):
                # find the index of those points in the landmarks + boundry array
                ind = (points.index([t[0], t[1]]), points.index([t[2], t[3]]), points.index([t[4], t[5]]))
                delaunay.append(ind)

        self._delaunay_triangles = delaunay

    def shape(self):
        """
        :return: the shape of the image
        """
        return self._image.shape[0:2]

    def image(self):
        return self._image.astype('uint8')

    def faces(self):
        return self._faces

    def landmarks(self):
        return self._landmarks

    def boundary(self):
        return self._boundary

    def points_of_interest(self):
        return np.concatenate((self._landmarks, self._boundary))

    def delaunay_triangles(self):
        return self._delaunay_triangles

    def point_in(self, point):
        if point[0] < 0:
            return False
        elif point[1] < 0:
            return False
        elif point[0] > self.shape()[1]:
            return False
        elif point[1] > self.shape()[0]:
            return False
        return True