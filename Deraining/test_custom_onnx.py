import cv2
import numpy as np
import onnx
import onnxruntime as ort

from skimage import img_as_ubyte


def preProcess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image


# video_path = "/home/ao/tmp/clip_videos/h97cam_water_video.mp4"
video_path = "/content/drive/MyDrive/DERAIN/DATA_captured/something_else/dusty_water_video1.mp4"
video = cv2.VideoCapture(video_path)


model_path = './checkpoints/mprnet.onnx'
ort_session = ort.InferenceSession(model_path)

while 1:
    ret, frame = video.read()
    if not ret:
        break
    input_image = preProcess(frame)
    raw = ort_session.run(None, {"input": input_image})
    pred = raw[0][0]
    print("[INFO] pred shape ", pred.shape)
    pred = img_as_ubyte(pred)
    pred = cv2.resize(pred, (frame.shape[1], frame.shape[0]))
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    image_show = np.concatenate((frame, pred), axis=1)
    cv2.imshow("image", image_show)
    if cv2.waitKey(1) == 27:
        break
