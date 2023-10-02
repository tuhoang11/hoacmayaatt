from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.models import load_model
from keras_visualizer import visualizer

model = load_model('mymodel.h5')
visualizer(model, format='png', view=True)