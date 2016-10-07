from flask import Flask, render_template, request
app = Flask(__name__)
import model
import json
import numpy as np

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/run/', methods=['POST'])
def run():
	if request.method == 'POST':
		training_data_indicies = json.loads(request.data)["training_data_indicies"]
		print(training_data_indicies)
		print(type(training_data_indicies))

		initial_image_indicies = json.loads(request.data)["initial_image_indicies"]
		print(initial_image_indicies)
		print(type(initial_image_indicies))

		number_of_times_clicked = json.loads(request.data)["number_of_times_clicked"]
		step_size = json.loads(request.data)["step_size"]

		model_type = json.loads(request.data)["model_type"]
		epoch = json.loads(request.data)["epoch"]

		results, errors = model.model(training_data_indicies, initial_image_indicies, number_of_times_clicked, step_size, model_type, epoch)

		results = results.tolist()
		errors = errors.tolist()

		return json.dumps({'results': results, 'errors': errors})

	return None


if __name__ == "__main__":
    app.debug = True
    app.run()

