import json
from flask import Flask, request, jsonify
from autocards import Autocards


app = Flask(__name__)
a = Autocards()


@app.route("/create_clozes", methods=["POST"])
def create_clozes():
    content = request.json
    clozes = a.create_clozes(content["text"])
    return jsonify(clozes)


@app.route("/create_qas", methods=["POST"])
def create_qas():
    content = request.json
    qas = a.create_qas(content["text"])
    return jsonify(qas)


@app.route("/reformulate_cloze", methods=["POST"])
def reformulate_cloze():
    content = request.json
    qa = a.create_clozes(content["cloze"])
    return jsonify(qa)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
