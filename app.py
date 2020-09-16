import os
from model import nmt_dt2ar

from flask import Flask, request, send_from_directory, render_template, json

app = Flask(__name__)

save_dir = '/app/model/model_save'

nmt = nmt_dt2ar()
nmt.load_model(save_dir=save_dir)
x = 'test'

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, app.static_folder), 'favicon.ico', mimetype='favicon.ico')



@app.route('/robots.txt')
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html", message='')



@app.route("/", methods=['POST'])
def move_forward():
    DT_txt = request.form.get('arabic-text')
    ara_txt = ''
    try:
       ara_txt,_,_ = nmt.translate_api_response(DT_txt)
    except KeyError as e:
        ara_txt = "Error with this word '%s', Check the spelling."%e
    finally:
        return render_template('index.html', message=[DT_txt, ara_txt])


app.config['TRAP_HTTP_EXCEPTIONS']=True
app.register_error_handler(Exception, defaultHandler)

"""
@app.route('/api', methods=['GET'])
def api():
    isDataComplete = request.args.get("q") and request.args.get("sourceLang") and request.args.get("targetLang")
    # check that all GET data need is received
    if isDataComplete:
        __GET__ = {"text": request.args.get("q"),
                   "src": request.args.get("sourceLang"),
                   "target": request.args.get("targetLang")}

        translated = "translate \" %s \" from %s to %s " % (__GET__["text"], __GET__["src"], __GET__["target"])
        translated_to, confidence, backend = nmt.translate_api_response(__GET__["text"])
        response_json = {
            "src": __GET__["src"],
            "confidence": 1,
            "sentences": [{
                            "trans": translated_to,
                            "orig": __GET__["text"],
                            "backend": 1}],
        }

        response = app.response_class(
            response=json.dumps(response_json),
            status=200,
            mimetype='application/json'
        )
        return response

    return render_template('error.html')
"""

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
