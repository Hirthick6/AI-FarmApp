from flask import Flask, render_template, request
import pickle
import sklearn
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('Home.html')


@app.route('/crop_recommend')
def crop_recommend():
    return render_template('crop.html')


@app.route('/fertilizer_recommend')
def fertilizer_recommend():
    return render_template('fertilizer.html')


@app.route('/crop_disease')
def crop_disease():
    return render_template('crop_disease.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/crop_predict', methods=['POST'])
def crop_predict():
    if request.method == 'POST':
        nitrogen = request.form['nitrogen']
        phosphorous = request.form['phosphorous']
        potassium = request.form['potassium']
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        ph = request.form['ph']
        rainfall = request.form['rainfall']

        model = pickle.load(open('Saved_models/crop_recommendation.pkl', 'rb'))
        prediction = model.predict(
            [[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]])
        text = "The best crop to cultivate in your land is "
        text = text + (''.join(prediction))

        return render_template('result.html', text=text)


@app.route('/fertilizer_predict', methods=['POST'])
def fertilizer_predict():
    if request.method == 'POST':
        temperature = request.form['temperature']
        humidity = request.form['humidity']
        moisture = request.form['moisture']
        soil = request.form['soil']
        crop = request.form['crop']
        nitrogen = request.form['nitrogen']
        phosphorous = request.form['phosphorous']
        potassium = request.form['potassium']

        d1 = {
            'Sandy': 1, 'Loamy': 2, 'Black': 3, 'Red': 4, 'Clayey': 5
        }

        d2 = {
            'Maize': 1, 'Sugarcane': 2, 'Cotton': 3, 'Tobacco': 4, 'Paddy': 5, 'Barley': 6,
            'Wheat': 7, 'Millets': 8, 'Oil seeds': 9, 'Pulses': 10, 'Ground Nuts': 11
        }

        soil = d1[soil]
        crop = d2[crop]

        model = pickle.load(
            open('Saved_models/fertilizer_recommendation.pkl', 'rb'))
        prediction = model.predict(
            [[temperature, humidity, moisture, soil, crop, nitrogen, phosphorous, potassium]])
        text = "The best fertilizer to use for your crop is "
        text = text + (''.join(prediction))
        return render_template('result.html', text=text)


classes = ['Tomato Late blight', 'Tomato healthy', 'Grape healthy',
           'Orange Haunglongbing (Citrus greening)', 'Soybean healthy', 'Squash Powdery mildew',
           'Potato healthy', 'Corn (maize) Northern Leaf Blight', 'Tomato Early blight',
           'Tomato Septoria leaf spot', 'Corn (maize) Cercospora leaf spot Gray leaf spot',
           'Strawberry Leaf scorch', 'Peach healthy', 'Apple Apple scab',
           'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Bacterial spot', 'Apple Black_rot',
           'Blueberry healthy', 'Cherry (including sour) Powdery mildew',
           'Peach Bacterial spot', 'Apple Cedar apple rust',
           'Tomato Target Spot', 'Pepper, bell healthy',
           'Grape Leaf blight (Isariopsis Leaf Spot)', 'Potato Late blight',
           'Tomato Tomato mosaic virus', 'Strawberry healthy', 'Apple healthy',
           'Grape Black rot', 'Potato Early blight', 'Cherry (including sour) healthy',
           'Corn (maize) Common rust ', 'Grape Esca (Black Measles)', 'Raspberry healthy',
           'Tomato Leaf Mold', 'Tomato Spider mites Two-spotted spider mite',
           'Pepper, bell Bacterial spot', 'Corn (maize) healthy']


@app.route('/crop_disease_predict', methods=['POST'])
def crop_disease_predict():
    if request.method == 'POST':
        files = request.files['crop_image']
        basepath = os.path.dirname(__file__)
        filename = files.filename
        filename = filename.replace(" ", "")
        file_path = os.path.join(basepath, 'uploads', filename)
        files.save(file_path)

        model_path = 'Saved_models/crop_disease_model'
        model = load_model(model_path)

        img = image.load_img(file_path, target_size=[224, 224])
        x = image.img_to_array(img)
        x = x/255
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        preds = np.argmax(preds)
        prediction = classes[preds]

        if 'healthy' in prediction:
            text = "Don't worry your crop is healthy"
        else:
            text = "Your crop is affected by the disease " + prediction
        return render_template('result.html', text=text)


if __name__ == '__main__':
    app.run(debug=False, host = "0.0.0.0")

