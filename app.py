from json import load
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import send_from_directory
from model.cnnmodel import predict
from model.model_init import load_cnn_model, load_text_model
from model.textmodel import get_sentiment

UPLOAD_FOLDER = 'static/files/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'csv', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_PATH'] = 2**16

global cnn_model, text_model, tokenizer
cnn_model = load_cnn_model()
text_model, tokenizer = load_text_model()

global sent
sent = {
    'positive' : 'позитивной',
    'negative' : 'негативной',
    'neutral'  : 'нейтральной'
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html', query=True)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            prediction = predict(cnn_model, filepath)
        else:
            return render_template('image.html', alert='Choose correct file!')
    return render_template('image.html', img=filepath, label=prediction)

@app.route('/analyzer', methods=['POST'])
def sent_analysis():
    if request.method == 'POST':
        text = request.form['sentiment']
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        result = get_sentiment(text_model, inputs)
        # print(result)

    return render_template('sentiment.html', result=sent[result], text=text)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 