# ShapeShop
*Towards Understanding Deep Learning Representations via Interactive Experimentation*

The Shape Workshop (**ShapeShop**) is an interactive system for visualizing and understanding what representations a neural network model has learned in images consisting of simple shapes. It encourages model building, experimentation, and comparison to helps users explore the robustness of image classifiers. 

[Read the paper][paper].  
[View the poster][poster].  
[Watch the teaser][teaser-acm].

***

![UI](images/github-ui-fig.png)

## Installation

We suggest creating a new environment to run ShapeShop. If you are using [Anaconda][anaconda], create a new environment called `shapeshop` by running

```bash
conda create --name shapeshop python=3
```

Switch to this environment by running

```bash
source activate shapeshop
```

### Requirements: Python (3.5)

From within the new environment install the following packages with the versions listed below:

numpy==1.11.3  
scipy==0.18.1  
flask==0.11.1  
keras==1.2.0  
tensorflow==0.12.1  
matplotlib==2.0.0  
Pillow==4.1.1  

For Keras, use our backend provided in `keras.json`. Your `shapeshop` environment's `keras.json` backend is located `$HOME/.keras/keras.json`. See [keras.io/backend][keras-backend] for more details.

### Requirements: JavaScript
D3 4.0 (loaded via web, no installation needed)  
jQuery 1.12.4 (loaded via web, no installation needed)

## Download or Clone

Once the requirements have been met, simply download or clone the repository. 

```bash
git clone https://github.com/fredhohman/shapeshop.git
```

## Usage

### Running ShapeShop

Run the system by 
```bash

python server.py
```
from the `shapeshop/` directory and pointing your browser to `http://localhost:5000`.

### Using ShapeShop

To use ShapeShop, follow the enumerated steps. 

1. **Select Training Data:** Choose what training data you want include. The number of training images chosen corresponds to how many classes the image classifier contains. You must select at least two (when two are chosen, this corresponds to binary classification)!
2. **Select Model:** Choose which model you want to use. MLP corresponds to a multilayer perceptron and CNN corresponds to a convolutional neural network.
3. **Select Hyperparameters:** Choose what hyperparameters you want for model training and the image generation process.
4. **Train and Visualize:** Click the button to train the model and generate your results!

ShapeShop uses the class activation maximization visualization technique to maximize each class to produce N images, each corresponding to one class. The system then presents all N resulting images, correlation coefficients, and the original class desired to be visualized back to the user for visual inspection and comparison. This process then repeats, where the user can select different images to train on, produce more images from new models, and compare to the previous results.

## Citation

**ShapeShop: Towards Understanding Deep Learning Representations via Interactive Experimentation.**  
Fred Hohman, Nathan Hodas, Duen Horng Chau.  
*Extended Abstracts, ACM Conference on Human Factors in Computing Systems (CHI). May 6-11, 2017. Denver, CO, USA.*

[Read the paper][paper].  
[View the poster][poster].  
Watch the teaser: [ACM][teaser-acm], [PoloClub][teaser-poloclub].

## License

MIT License. See [`LICENSE.md`](LICENSE.md).

## Credits 

For questions and support contact [Fred Hohman][fred].

[anaconda]: https://www.continuum.io/downloads
[keras-backend]: https://keras.io/backend/
[paper]: http://www.cc.gatech.edu/~dchau/papers/17-chi-shapeshop.pdf
[poster]: http://fredhohman.com/assets/shapeshop-poster.pdf
[teaser-poloclub]: https://youtu.be/XL3fFwZ1DDQ
[teaser-acm]: https://www.youtube.com/watch?v=wRX1xEdrD1g
[fred]: http://www.fredhohman.com
