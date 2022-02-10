from skimage import feature
import numpy as np


class LocalBinaryTOP:
    def __init__(self, numPoints, radius, scale):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        self.scale = scale

    def describe(self, middle_frame, front_frame, back_frame, eps=1e-7):
        xyHist = []
        for y in range(0, len(middle_frame), self.scale):
            for x in range(0, len(middle_frame[0]), self.scale):
                submatrix = [i[x:x + self.scale] for i in middle_frame[y:y + self.scale]]
                lbp = feature.local_binary_pattern(submatrix, self.numPoints,
                                                   self.radius)
                (hist, _) = np.histogram(lbp.ravel(),
                                         bins=np.arange(257))

                xyHist += hist.tolist()
        valuesForXT = []
        for rowNumber in range(0, len(middle_frame)):
            submatrix = [front_frame[rowNumber]] + [middle_frame[rowNumber]] + [back_frame[rowNumber]]
            lbp = feature.local_binary_pattern(submatrix, self.numPoints,
                                               self.radius)
            valuesForXT += list(lbp.ravel())
        (xtHist, _) = np.histogram(valuesForXT, bins=np.arange(257))
        valuesForYT = []
        for columnNumber in range(0, len(middle_frame[0])):
            middleC = [line[columnNumber] for line in middle_frame]
            frontC = [line[columnNumber] for line in front_frame]
            backC = [line[columnNumber] for line in back_frame]
            submatrix = [[] for i in range(len(middle_frame))]
            for rowNumber in range(len(middle_frame)):
                submatrix[rowNumber] = [frontC[rowNumber], middleC[rowNumber], backC[rowNumber]]
            lbp = feature.local_binary_pattern(submatrix, self.numPoints,
                                               self.radius)
            valuesForYT += list(lbp.ravel())
        (xtHist, _) = np.histogram(valuesForXT, bins=np.arange(257))
        (ytHist, _) = np.histogram(valuesForYT, bins=np.arange(257))
        grandHist = xyHist + xtHist.tolist() + ytHist.tolist()
        return grandHist
