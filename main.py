from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import io

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from Azure!"}

@app.post("/upload-csv/")
async def create_plot(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    expected_columns = ["timestamp", "acc_x", "acc_y", "acc_z",
                        "gyro_x", "gyro_y", "gyro_z",
                        "rot_w", "rot_x", "rot_y", "rot_z"]

    if not all(col in df.columns for col in expected_columns):
        return {"error": "CSV doesn't contain required columns."}

    # Генерируешь изображение и отправляешь в память (без сохранения)
    img_bytes = plot_trajectory_with_orientation(df)

    # Отправляешь сразу в ответе
    return StreamingResponse(img_bytes, media_type="image/png")

def plot_trajectory_with_orientation(data):
    rotation_vector = data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    positions = [np.array([0.0, 0.0, 0.0])]
    orientations = []
    step_size = 0.01

    for quat in rotation_vector:
        if np.isnan(quat).any():
            orientations.append(np.eye(3))
            positions.append(positions[-1])
            continue
        r = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
        rot_matrix = r.as_matrix()
        orientations.append(rot_matrix)
        forward = rot_matrix[:, 0]
        positions.append(positions[-1] + step_size * forward)

    positions = np.array(positions[1:])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color='black', label='Phone trajectory')

    scale = 0.01
    for i in range(0, len(positions), 5):
        p = positions[i]
        rot = orientations[i]
        ax.quiver(p[0], p[1], p[2], rot[0, 0], rot[1, 0], rot[2, 0], color='r', length=scale)
        ax.quiver(p[0], p[1], p[2], rot[0, 1], rot[1, 1], rot[2, 1], color='g', length=scale)
        ax.quiver(p[0], p[1], p[2], rot[0, 2], rot[1, 2], rot[2, 2], color='b', length=scale)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Phone trajectory with orientation axes")
    ax.legend()
    plt.tight_layout()

    # Сохраняешь изображение в памяти
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    img_buf.seek(0)

    return img_buf
