import datetime
import json
import os

import cv2
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from PIL import Image
from werkzeug.utils import secure_filename

from serving import Serving


app = Flask(__name__)
app.config.from_pyfile('config.py')

app.secret_key = app.config['KEY']
app.permanent_session_lifetime = datetime.timedelta(minutes=10)


@app.route('/')
def root():
    return redirect(url_for('inference'))


@app.route('/inference', methods=['GET', 'POST'])
def inference():
    if request.method == 'POST':
        # Option1: image file upload
        if 'file' in request.files:
            image = request.files['file']
            if image.filename == '':
                print('No file attachemnt')
                flash('Error: File is not attached')
                return redirect(request.url)

            if image.filename.split('.')[-1].lower() not in app.config['EXT']:
                print('No image file')
                flash('Error: The attached file is not image file')
                return redirect(request.url)

            filename = secure_filename(image.filename)
            dir_name = datetime.datetime.now().strftime('%Y%m%d%H%M')
            image_data = Image.open(image)

        # Option2: sample image selection
        elif 'select' in request.form:
            filename = request.form['select']
            dir_name = datetime.datetime.now().strftime('%Y%m%d%H%M') + '_sample'
            image_data = Image.open(
                os.path.abspath(os.path.join(app.config['PATH_SAMPLE_DATA'], filename))
            )

        else:
            print('Sample image is not selected')
            flash('Error: Sample image is not selected')
            return redirect(request.url)

        # Save uploaded or selected file
        path = os.path.abspath(os.path.join(app.config['PATH_DATA'], dir_name))
        os.mkdir(path)
        image_data.save(os.path.join(path, filename))
        inference_filename = 'inference_' + filename

        # Machine learning inference
        serving = Serving(image_data)
        serving.load_model()
        serving.transform()
        output_image, output_score = serving.execution()

        # Save output image and data
        cv2.imwrite(os.path.join(path, inference_filename), output_image)
        with open(os.path.join(path, 'inference_output.json'), 'w') as f:
            json.dump(output_score, f)

        return render_template(
            'inference_result.html', dir=dir_name, file=inference_filename
        )

    # GET route (display initial screen)
    sample_list = os.listdir(os.path.abspath(app.config['PATH_SAMPLE_DATA']))
    return render_template('inference_init.html', list=sample_list)


@app.route('/data/<dir>/<file>')
def path_image(dir, file):
    return send_from_directory(
        os.path.abspath('./data/{}'.format(dir)), file, as_attachment=True
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
