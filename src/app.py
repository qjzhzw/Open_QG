import sys
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from flask import Flask, url_for, redirect, render_template, request
app = Flask(__name__)

from demo import demo, init


@app.route('/query', methods=['GET', 'POST'])
def interact():
    print(dir(request))
    if request.method == 'POST':
        data = request.get_data().decode("utf-8")
        data = json.loads(data)
        sentence = data['sentence']
        answer = data['answer']
    else:
        sentence = 'There are 5000000 people in the united states .'
        answer = '5000000'
    question = demo(sentence, answer, logger, params, vocab, model, generator)
    ret = json.dumps({'return':question})
    return ret


if __name__ == '__main__':
    logger, params, vocab, model, generator = init()
    app.run(host="127.0.0.1", port=8091, debug=False, use_reloader=False)
