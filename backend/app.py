from backend.model_utils import get_model, predict_img
from flask import Flask, request, render_template, send_file,jsonify
from PIL import Image
import numpy as np
import torchvision
import io
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

app = Flask(__name__)


# Route to render the upload form template
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route to handle image uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        try:
            # Read image file
            img = Image.open(file)
            img = img.convert('RGB')
            img = torchvision.transforms.functional.pil_to_tensor(img)/255.

            model_name = request.form['model']
            # Load selected model
            print(model_name)
            derain_model = get_model(model_name)
            derain_model.eval()
            # print(img.shape)
            derained_img = predict_img(derain_model, img, model_name=model_name)

            torch.save({
                "img": derained_img,
            }, "img.pt")
            derained_img = derained_img.squeeze().detach()

            # Save processed image temporarily
            processed_img_path = 'processed_image.jpg'
            save_image(derained_img, processed_img_path)

            # Redirect to the processed image template
            return render_template('processed.html')

        except Exception as e:
            return jsonify({'error': str(e)})

    return jsonify({'error': 'Invalid file format'})


# Route to serve the processed image
@app.route('/processed_image')
def processed_image():
    return send_file('processed_image.jpg', mimetype='image/jpg')


def allowed_file(filename):
    # You can define your own criteria for allowed file formats here
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}


if __name__ == '__main__':
    app.run(debug=True)