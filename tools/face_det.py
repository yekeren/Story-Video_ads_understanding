import sys

import numpy as np
import openface

dlibFacePredictorPath = 'openface/models/dlib/shape_predictor_68_face_landmarks.dat'
align = openface.AlignDlib(dlibFacePredictorPath)

assert align is not None


bbox = align.getAllFaceBoundingBoxes(np.zeros((300, 300, 3), dtype=np.uint8))

print bbox
print >> sys.stderr, 'Done'

