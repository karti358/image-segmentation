from pathlib import Path
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

import sys
video_path = str(sys.argv[0])
save_path = str(sys.argv[1])
print(video_path , save_path)

import cv2

cap = cv2.VideoCapture(str(video_path))
if cap.isOpened():
    print("Cap is successfully created")


this_file = Path.cwd()
print(this_file)

my_model = tf.keras.models.load_model(this_file.joinpath(
    'models/model_v2.h5').as_posix(), compile=False)


def showing_predictions(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (96, 128), method='nearest')
    image = image[tf.newaxis, ...]
    pred_mask = my_model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = tf.squeeze(pred_mask, axis=0)
    pred_mask = tf.cast(pred_mask, dtype=tf.uint8)
    pred_mask = tf.keras.utils.array_to_img(pred_mask)
    return np.array(pred_mask)

def grab_frame():
    ret, frame = cap.read()
    if ret == False:
        return []
    cv2.imshow('Image', frame)
    seg_img = showing_predictions(frame)
    seg_img = cv2.resize(seg_img, (0, 0), fx=5, fy=5)

    if cv2.waitKey(1) == ord('x'):
        cap.release()
        cv2.destroyAllWindows()
        return []
    return seg_img

fig, ax = plt.subplots()
im = ax.imshow(grab_frame())


def update(frame):
    im.set_data(grab_frame())


anim = animation.FuncAnimation(plt.gcf(), update, interval=200)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5)
try:
    anim.save(save_path, writer=writer)
except TypeError:
    # writer.close()
    anim.event_source.stop()
