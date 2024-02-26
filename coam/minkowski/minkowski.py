import os
import sys

import numpy as np

# noinspection PyUnresolvedReferences
import pymesh

# while os.path.exists("tmp/optimize.coam"):
mesh = pymesh.load_mesh(f"/tmp/iterations/{sys.argv[3]}/{sys.argv[1]}")
mesh = pymesh.minkowski_sum(
    mesh, np.array([[float(sys.argv[2]), 0, 0], [-float(sys.argv[2]), 0, 0]])
)
pymesh.save_mesh(
    f"/tmp/iterations/{sys.argv[3]}/{os.path.splitext(sys.argv[1])[0]}_minkowski.stl",
    mesh,
)
