"""Flask server that transfers data between the ShapeShop UI and `model.py`.
"""

import model
import json
from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run/', methods=['POST'])
def run():
    if request.method == 'POST':
        training_data_indicies = json.loads(request.data.decode('utf-8'))["training_data_indicies"]
        print(training_data_indicies)
        print(type(training_data_indicies))

        initial_image_indicies = json.loads(request.data.decode('utf-8'))["initial_image_indicies"]
        print(initial_image_indicies)
        print(type(initial_image_indicies))

        number_of_times_clicked = json.loads(request.data.decode('utf-8'))["number_of_times_clicked"]
        step_size = json.loads(request.data.decode('utf-8'))["step_size"]

        model_type = json.loads(request.data.decode('utf-8'))["model_type"]
        epoch = json.loads(request.data.decode('utf-8'))["epoch"]

        results, errors, training_data_indicies_nonzero = model.model(training_data_indicies,
                                                                      initial_image_indicies,
                                                                      number_of_times_clicked,
                                                                      step_size,
                                                                      model_type,
                                                                      epoch)

        results = results.tolist()
        errors = errors.tolist()
        training_data_indicies_nonzero = training_data_indicies_nonzero.tolist()

        return json.dumps({'results': results,
                           'errors': errors,
                           'training_data_indicies_nonzero': training_data_indicies_nonzero})

    return None


if __name__ == "__main__":
    app.debug = True
    app.run()
