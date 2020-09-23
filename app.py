import os
from model import nmt_dt2ar

from flask import Flask, request, send_from_directory, render_template, json

app = Flask(__name__)

save_dir = 'C:/Users/hp/Desktop/internship2020/NMT att/machine-translation-nmt/model/model_save'


nmt = nmt_dt2ar()
nmt.load_model(save_dir=save_dir)
x = 'test'

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, app.static_folder), 'favicon.ico', mimetype='favicon.ico')





@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")





@app.route('/Translate', methods = ['POST','GET'])
def Translate():
    
        if request.method == "POST":
            DT=request.form["message"]
            AR=nmt.translate(message)
            return render_template('index.html', message=DT, output=AR)
        else:
            return render_template('index.html')








if __name__ == '__main__':
    app.run(debug=True)
    app.config[SERVER_NAME]='localhost:5000'
