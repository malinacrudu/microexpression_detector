from skimage import feature
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius, scale):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        self.scale = scale

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        grand_hist = []
        for y in range(0, len(image), self.scale):
            for x in range(0, len(image[0]), self.scale):
                submatrix = [i[x:x + self.scale] for i in image[y:y + self.scale]]
                lbp = feature.local_binary_pattern(submatrix, self.numPoints,
                                                   self.radius)
                (hist, _) = np.histogram(lbp.ravel(),
                                         bins=np.arange(257))

                grand_hist += hist.tolist()
        # # normalize the histogram
        # hist = hist.astype("float")
        # hist /= (hist.sum() + eps)
        # # return the histogram of Local Binary Patterns
        return grand_hist


