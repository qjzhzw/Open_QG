import sys
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from flask import Flask, url_for, redirect, render_template, request
app = Flask(__name__)

from src.API import demo

@app.route('/query', methods=['POST'])
def interact():
    data = request.get_data().decode("utf-8")
    data = json.loads(data)
    sentence = data['sentence']
    answer = data['answer']
    # sentence = 'what'
    # answer = 'what'
    question = demo(sentence, answer)
    ret = json.dumps({'return':question})
    return ret


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8091, debug=False, use_reloader=False)
