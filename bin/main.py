import os
import pickle
import neat
import pygame
import time
import matplotlib.pyplot as plt
from environment import Bird, Environment, collision, WIN_HEIGHT
from visualize import nn_visualize

# ? HYPERPARAM

VISUALIZE_NN = True

LIMIT_FRAME = True
SPEED_START = 100
SPEED_END = 100

##################

if __name__ == '__main__':
    main_dir = os.path.dirname(os.getcwd())
    model_dir = os.path.join(main_dir, 'model')
    config_file = os.path.join(main_dir, 'neat-config.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    model = None
    with open(os.path.join(model_dir, 'best_bird.pickle'), 'rb') as x:
        model = pickle.load(x)
    model = neat.nn.FeedForwardNetwork.create(model, config)
    model_layers = [3, 1]
    model_weights = [[-0.04946766217553714], [1.4825914566093632], [-1.6677928227356724]]
    model_biases = [-1.7508011381023019]
    if LIMIT_FRAME:
        clock = pygame.time.Clock()
    while True:
        bird = Bird(300)
        env = Environment()
        done = False
        tick = SPEED_START
        debug_time = 0
        
        while not done:
            try:
                birds = [bird]
                env.play_step(birds)
                front_pipe = None
                for pipe in env.pipes:
                    if pipe.front == True:
                        front_pipe = pipe
                        break
                assert(front_pipe is not None)
                nn_input = (bird.y, abs(front_pipe.upper_bottom-bird.y), abs(front_pipe.lower_up-bird.y))
                output = model.activate(nn_input)
                if VISUALIZE_NN:
                    out = nn_visualize(model_layers, model_weights, model_biases, nn_input, output)
                if output[0] > 0.5:
                    bird.jump()
                for pipe in env.pipes:
                    if collision(pipe, bird) or bird.y < 0 or bird.y > WIN_HEIGHT:
                        done = True
                        break
                env.render(birds)
                if env.score%100 == 0 and env.score is not 0 and env.score is not debug_time:
                    print(f"Score = {env.score} \t Tick = {tick}")
                    debug_time = env.score
                if LIMIT_FRAME:
                    clock.tick(round(tick))
                if tick < SPEED_END:
                    tick += .1
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONUP:
                        None
            except KeyboardInterrupt:
                done = True
        break