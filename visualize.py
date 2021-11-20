import numpy as np
import cv2
import math
import os
import pickle
import neat
import random
from PIL import Image, ImageDraw

WIN_WIDTH = 600
WIN_HEIGHT = 400

# ? Layer represent as a 1 Dimensional List Define its Neuron Count
# ? Weights is encoded as a 3 Dimensional List sorted by First Layer's weights to Next Layer (Tensorflow-Like)
# ? Biases is encoded as a 1 Dimensional List sorted from first hidden layer's neuron into last output layer's
def nn_visualize(layers, weights, biases, inputs):
    neuron_size = 30
    padding_x = 150

    img_array = np.zeros((3000, 3000, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img = img.resize((WIN_WIDTH, WIN_HEIGHT), Image.ANTIALIAS)
    imdraw = ImageDraw.Draw(img)
    neuron_pos = [[]]
    x = padding_x
    delta_x = ((WIN_WIDTH-2*padding_x)-neuron_size*len(layers))//(len(layers)-1)
    # ? Generation Neuron Coordinate
    for inx, layer in enumerate(layers):
        delta_y = (WIN_HEIGHT-(neuron_size*layer))//(layer+1)
        y = delta_y
        neuron_pos.append([])
        for neuron in range(0, layer):
            # imdraw.ellipse([(x, y), (x+neuron_size, y+neuron_size)], (255, 255, 255))
            neuron_pos[inx].append(((2*x+neuron_size)//2, (2*y+neuron_size)//2))
            y += delta_y
            y += neuron_size
        x += delta_x
        x += neuron_size
    assert(len(neuron_pos[-1]) == 0)
    neuron_pos.pop(-1)
    # ? Rendering Weights
    neuron_bf = [X for X in inputs]
    neuron_af = []
    bias_cnt = 0
    neuron_activated = np.zeros((100, 100), dtype=bool)
    for layer in range(0, len(neuron_pos)-1):
        for inxA, neuron1 in enumerate(neuron_pos[layer]):
            temp = bias_cnt
            for inxB, neuron2 in enumerate(neuron_pos[layer+1]):
                neuron_val = biases[temp]+neuron_bf[inxA]*weights[inxA][inxB]
                if len(neuron_af) > inxB:
                    neuron_af[inxB] += neuron_val
                else:
                    neuron_af.append(neuron_val)
                if weights[inxA][inxB] >= 0:
                    imdraw.line([(neuron1[0], neuron1[1]), (neuron2[0], neuron2[1])], (0, 255, 0), 3)
                else:
                    imdraw.line([(neuron1[0], neuron1[1]), (neuron2[0], neuron2[1])], (0, 0, 255), 3)
                temp += 1
        neuron_bf = neuron_af
        bias_cnt += layers[layer+1]
        # ! Activation Function
        for inx, neuron_val in enumerate(neuron_bf):
            neuron_bf = np.tanh(neuron_val)
            neuron_activated[layer+1] = neuron_val > 0.5
        neuron_af = []

    # ? Rendering Neuron
    for inxA, layer in enumerate(neuron_pos):
        for inxB, neuron in enumerate(layer):
            if neuron_activated[inxA][inxB]:
                imdraw.ellipse([(neuron[0]-neuron_size//2, neuron[1]-neuron_size//2), (neuron[0]+neuron_size//2, neuron[1]+neuron_size//2)], (255, 255, 255))
            else:
                imdraw.ellipse([(neuron[0]-neuron_size//2, neuron[1]-neuron_size//2), (neuron[0]+neuron_size//2, neuron[1]+neuron_size//2)], (0, 0, 255))
    cv2.imshow('img', np.array(img))
    cv2.moveWindow('img', 935, 0)
    return neuron_bf

if __name__ == '__main__':
    main_dir = os.path.dirname(os.getcwd())
    model_dir = os.path.join(main_dir, 'model')
    config_dir = os.path.join(main_dir, 'neat-config.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_dir)
    model = None
    with open(os.path.join(model_dir, 'best_bird.pickle'), 'rb') as X:
        model = pickle.load(X)
    
    model_net = neat.nn.FeedForwardNetwork.create(model, config)
    a, b, c = random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)
    output = model_net.activate((a, b, c))
    my_output = nn_visualize([3, 1], [[-0.04946766217553714], [1.4825914566093632], [-1.6677928227356724]], [-1.7508011381023019], (a, b, c))
    print(abs(output[0]-my_output))
