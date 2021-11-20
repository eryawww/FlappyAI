import os
from environment import Bird, collision, Environment, WIN_HEIGHT
import neat
import pygame
import pickle

TRAINING = False

gen = 0
def evolve(genomes, config):
    global gen
    nets = []
    ge = []
    birds = []
    for g_id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(300))
        g.fitness = 0
        ge.append(g)
    env = Environment()
    done = False
    clock = pygame.time.Clock()
    while not done:
        env.play_step(birds)
        for x, bird in enumerate(birds):
            front_pipe = None
            for pipe in env.pipes:
                if pipe.front == True:
                    front_pipe = pipe
                    break
            assert(front_pipe is not None)
            ge[x].fitness += .1
            output = nets[x].activate((bird.y, abs(bird.y-front_pipe.upper_bottom), abs(bird.y-front_pipe.lower_up)))
            if output[0] > 0.5:
                bird.jump()
        for bird in birds:
            for pipe in env.pipes:
                if collision(pipe, bird) or bird.y > WIN_HEIGHT or bird.y < 0:
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))
        if len(birds) == 0:
            break
        if ge[0].fitness > 100:
            break
        env.render(birds, gen)
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                None
        
        clock.tick(30)
    gen += 1

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, 
                                neat.DefaultStagnation, config_path)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    
    winner = population.run(evolve)

    if TRAINING:
        dest_dir = os.getcwd()
        dest_dir = os.path.dirname(dest_dir)
        dest_dir = os.path.join(dest_dir, 'model')
        file_name = "best_bird.pickle"
        with open(os.path.join(dest_dir, file_name), 'wb') as x:
            pickle.dump(winner, x)
            
if __name__ == "__main__":
    local_dir = os.getcwd()
    config_path = os.path.dirname(local_dir)
    config_path = os.path.join(config_path, 'neat-config.txt')
    run(config_path)

    