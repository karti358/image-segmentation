from pathlib import Path
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import cv2
from pathlib import Path

this_file = Path.cwd()
UPLOAD_FOLDER = this_file.joinpath('static/uploaded_videos')
SEGMENTED_FOLDER = this_file.joinpath('static/segmented_videos')

my_model = tf.keras.models.load_model(this_file.joinpath(
    'models/model_v2.h5').as_posix(), compile=False)


def get_predictions(video_path, save_path):
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

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        print("Cap is successfully created")

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

import sys
from flask import Flask, render_template, url_for, request, redirect, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER.as_posix()
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER.as_posix()


@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/', methods = ['POST'])
def save_uploaded_video():
    print("In upload function")
    if 'file' not in request.files:
        print(request.files)
        flash("No file given.")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Give valid video.')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)

        present_upload_path = UPLOAD_FOLDER.joinpath(filename).as_posix()
        present_save_path = SEGMENTED_FOLDER.joinpath(filename).as_posix()

        file.save(present_upload_path)

        flash('Vido uploaded successfully.')
        
        get_predictions(present_upload_path, present_save_path)

        return render_template('upload.html', filename = filename)

@app.route('/display_uploaded/<filename>')
def display_uploaded(filename):
    return redirect(url_for('static', filename = 'uploaded_videos/' + filename), code = 301)

@app.route('/display_segmented/<filename>')
def display_segmented(filename):
    return redirect(url_for('static', filename = 'segmented_videos/' + filename), code = 301)


if __name__ == "__main__":
    app.run(debug = True)


    
        