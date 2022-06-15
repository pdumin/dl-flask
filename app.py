from json import load
import os
from flask import Flask, redirect, render_template, request, flash, url_for
from werkzeug.utils import secure_filename
from model.cnnmodel import predict
from model.model_init import load_cnn_model, load_text_model
from model.textmodel import get_sentiment
import concurrent

UPLOAD_FOLDER = 'static/files/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_PATH'] = 2**16
app.secret_key = b'0000'

global sent
sent = {
    'positive' : 'позитивной',
    'negative' : 'негативной',
    'neutral'  : 'нейтральной'
}

# @app.before_first_request
# def init_models():
#     global cnn_model, text_model, tokenizer
#     cnn_model = load_cnn_model()
#     text_model, tokenizer = load_text_model()
#     thread = threading.Thread(target=init_models)
#     thread.start()

@app.before_first_request
def init_models():
    # задаем глобальные переменные, в которые запишем модели
    global cnn_model, text_model, tokenizer, text_generator_model, text_generator_tokenizer

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # создаем несколько потоков
        cnn_future = executor.submit(load_cnn_model)
        text_model_future = executor.submit(load_text_model)
        
        # забираем результат из каждого потока 
        cnn_model = cnn_future.result()
        text_model, tokenizer = text_model_future.result()


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

@app.route('/task')
def task():
    return render_template('task.html')

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
            flash('Choose correct file!')
            return redirect(url_for('image'))
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