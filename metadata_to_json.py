import pandas as pd
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil
from PIL import Image


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


metadata = pd.read_csv("~/testtank/bottom_camera_metadata.csv")

frames = [0]*len(metadata.index)
framestest = [0]*int(len(metadata.index)/8+1)
for i, row in metadata.iterrows():
    im = Image.open(str("/home/naglak/testtank/raw/"+row.filename))
    im.save(str("/home/naglak/testtank/train/"+row.filename.split(".tif")[0]+".png"))
    frames[i] = {"file_path": str("./train/"+row.filename.split(".tif")[0]), "rotation": 0.012566, "transform_matrix": np.concatenate((np.concatenate((R.from_quat([row.quat_z, row.quat_y, row.quat_z, row.quat_w]).as_matrix(), np.array([[row.pos_n, row.pos_e, row.pos_d]]).T), axis=1), [[0, 0, 0, 1]]), axis=0)}
    if not i%8:
        framestest[int(i/8)] = {"file_path": str("./test/"+row.filename.split(".tif")[0]), "rotation": 0.012566, "transform_matrix": np.concatenate((np.concatenate((R.from_quat([row.quat_z, row.quat_y, row.quat_z, row.quat_w]).as_matrix(), np.array([[row.pos_n, row.pos_e, row.pos_d]]).T), axis=1), [[0, 0, 0, 1]]), axis=0)}
        shutil.copy(str("/home/naglak/testtank/train/"+row.filename.split(".tif")[0]+".png"),str("/home/naglak/testtank/test/"+row.filename.split(".tif")[0]+".png"))

with open("/home/naglak/testtank/transforms_train.json", "w") as write_file:
    numpy_data = {"camera_angle_x": np.array([0.691]), "frames": frames}

    json.dump(numpy_data, write_file, indent=4, cls=NumpyArrayEncoder)

with open("/home/naglak/testtank/transforms_test.json", "w") as write_file:
    numpy_data = {"camera_angle_x": np.array([0.691]), "frames": framestest}

    json.dump(numpy_data, write_file, indent=4, cls=NumpyArrayEncoder)
