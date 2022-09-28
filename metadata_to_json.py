import pandas as pd
import json
import numpy as np
from scipy.spatial.transform import Rotation as R


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


metadata = pd.read_csv("~/test/bottom_camera_metadata.csv")

frames = [0]*len(metadata.index)
for i, row in metadata.iterrows():
    frames[i] = {"file_path": row.filename, "rotation": 0.012566, "transform_matrix": np.concatenate((np.concatenate((R.from_quat([row.quat_z, row.quat_y, row.quat_z, row.quat_w]).as_matrix(), np.array([[row.pos_n, row.pos_e, row.pos_d]]).T), axis=1), [[0, 0, 0, 1]]), axis=0)}

with open("/home/naglak/test/transforms_train.json", "w") as write_file:
    numpy_data = {"camera_angle_x": np.array([0.691]), "frames": frames}

    json.dump(numpy_data, write_file, cls=NumpyArrayEncoder)
