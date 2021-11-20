import os
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{340}, {30}"
import pygame
import random
import time
import math
import matplotlib.pyplot as plt

WIN_WIDTH = 600
WIN_HEIGHT = 800
DEBUG = True

# ? Horizontal Velocity
GAME_VEL = 7

pygame.init()
dir_img = 'D:/Code/Py/tensorflow/Genetic Algorithm/flappybird/img'
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
IMG_BASE = pygame.transform.scale2x(pygame.image.load(os.path.join(dir_img, 'base.png')).convert_alpha())
IMG_BIRD = [pygame.transform.scale2x(pygame.image.load(os.path.join(dir_img, f'bird{x}.png'))) for x in range(1, 4)]
IMG_PIPE_UP = pygame.transform.flip(pygame.transform.scale2x(pygame.image.load(os.path.join(dir_img, 'pipe.png')).convert_alpha()), False, True) 
IMG_PIPE_DOWN = pygame.transform.scale2x(pygame.image.load(os.path.join(dir_img, 'pipe.png')).convert_alpha())
IMG_BG = pygame.transform.scale(pygame.image.load(os.path.join(dir_img, 'bg.png')).convert_alpha(), (600, 900))

font = pygame.font.SysFont('Calibri Bold', 35)

class Base:
    def __init__(self):
        self.y = WIN_HEIGHT-IMG_BASE.get_height()+150
        self.x1 = 0
        self.x2 = IMG_BASE.get_width()

    def move(self):
        self.x1 -= GAME_VEL
        self.x2 -= GAME_VEL
        if self.x1+IMG_BASE.get_width() < 0:
            self.x1 = self.x2+IMG_BASE.get_width()
        if self.x2+IMG_BASE.get_width() < 0:
            self.x2 = self.x1+IMG_BASE.get_width()

class Pipe:
    def __init__(self):
        self.x = WIN_WIDTH
        self.front = True
        self.gap = 200
        self.upper_bottom = random.randint(0, WIN_HEIGHT-self.gap-100)
        self.lower_up = self.upper_bottom+self.gap
        self.upper_up = self.upper_bottom-IMG_PIPE_DOWN.get_height()
    
    def move(self):
        self.x -= GAME_VEL

class Bird:
    x = 100
    GRAVITY = 1
    MAX_GREVITATIONAL_PULL = 10
    JUMP_FORCE = 15
    def __init__(self, y):
        self.v = 0
        self.tick = 0
        self.rotation = 0
        self.y = y
        self.dis_img = 0
    
    def move(self):
        # ? HORIZONTAL COMPONENT
        self.tick += 1
        if self.tick % 5 == 0:
            self.dis_img += 1
            self.dis_img %= 3
        self.tick %= 1000
        # ? VERTICAL COMPONENT
        if self.v > 0:
            self.rotation -= 5
            self.rotation = max(-90, self.rotation)
            self.v = min(self.v, self.MAX_GREVITATIONAL_PULL) # ! MAX of GRAVITATIONAL PULL
        self.v += self.GRAVITY # ! GRAVITATIONAL PULL
        self.y += self.v
        
    def jump(self):
        self.v = -self.JUMP_FORCE
        self.rotation = 45

def blitRotateCenter(surf, img, topleft, angle):
    rotated_img = pygame.transform.rotate(img, angle)
    new_rect = rotated_img.get_rect(center = img.get_rect(topleft = topleft).center)

    surf.blit(rotated_img, new_rect.topleft)

class Environment:
    # ! BIRD IS NOT CARRIED BY THE ENVIROMENT, BIRD IS AN AGENT
    def __init__(self):
        self.pipes = [Pipe()]
        self.base = Base()
        self.win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        self.score = 0
        self.series = []
        self.time = time.time()

    def play_step(self, birds):
        for bird in birds:
            bird.move()
        for pipe in self.pipes:
            pipe.move()
        if birds[0].x > self.pipes[0].x+IMG_PIPE_DOWN.get_width() and self.pipes[0].front == True:
            self.pipes.append(Pipe())
            self.score += 1
            self.series.append(time.time()-self.time)
            self.time = time.time()
            self.pipes[0].front = False
        if self.pipes[0].x < 0-IMG_PIPE_DOWN.get_width():
            self.pipes.pop(0)
        
        self.base.move()
    
    def render(self, birds, gen = None):
        self.win.blit(IMG_BG, (0, 0))
        for bird in birds:
            blitRotateCenter(self.win, IMG_BIRD[bird.dis_img], (bird.x, bird.y), bird.rotation)
            front_pipe = None
            for pipe in self.pipes:
                if pipe.front == True:
                    front_pipe = pipe
                    break
            assert(front_pipe is not None)
            if DEBUG:
                bird_center = (bird.x+IMG_BIRD[0].get_width()/2, bird.y+IMG_BIRD[0].get_height()/2)
                pygame.draw.line(self.win, (255, 0, 0), bird_center, (front_pipe.x+IMG_PIPE_DOWN.get_width(), front_pipe.upper_bottom))
                pygame.draw.line(self.win, (255, 0, 0), bird_center, (front_pipe.x+IMG_PIPE_DOWN.get_width(), front_pipe.lower_up))
            
        for pipe in self.pipes:
            self.win.blit(IMG_PIPE_UP, (pipe.x, pipe.upper_up))
            self.win.blit(IMG_PIPE_DOWN, (pipe.x, pipe.lower_up))
        score_font = font.render(f'SCORE  {self.score}', 1, (255, 255, 255))
        self.win.blit(IMG_BASE, (self.base.x1, self.base.y))
        self.win.blit(IMG_BASE, (self.base.x2, self.base.y))
        if gen is not None:
            gen_font = font.render(f'GEN  {gen}', 1, (255, 255, 255))
            self.win.blit(gen_font, (WIN_WIDTH-gen_font.get_width()-5, 5))
        self.win.blit(score_font, (5, 5))

        pygame.display.update()

def play_step(bird, pipes, base):
    bird.move()
    for pipe in pipes:
        if collision(pipe, bird):
            return True
        pipe.move()
    base.move()
    if bird.x > pipes[0].x and pipes[0].front == True:
        pipes.append(Pipe())
        pipes[0].front = False
    if pipes[0].x < 0-IMG_PIPE_DOWN.get_width()-100:
        pipes.pop(0)
    return False

def render(win, bird:Bird, pipes, base:Base):
    win.blit(IMG_BG, (0, 0))
    win.blit(IMG_BASE, (base.x1, base.y))
    win.blit(IMG_BASE, (base.x2, base.y))
    blitRotateCenter(win, IMG_BIRD[bird.dis_img], (bird.x, bird.y), bird.rotation)
    for pipe in pipes:
        win.blit(IMG_PIPE_UP, (pipe.x, pipe.upper_up))
        win.blit(IMG_PIPE_DOWN, (pipe.x, pipe.lower_up))
    # win.blit(IMG_PIPE, (0, 0)) 
    pygame.display.update()

def collision(pipe, bird):
    bird_mask = pygame.mask.from_surface(IMG_BIRD[bird.dis_img])
    upperPipe_mask = pygame.mask.from_surface(IMG_PIPE_UP)
    lowerPipe_mask = pygame.mask.from_surface(IMG_PIPE_DOWN)
    upper_offset = (pipe.x-bird.x, pipe.upper_up-round(bird.y))
    lower_offset = (pipe.x-bird.x, pipe.lower_up-round(bird.y))

    u_collision = bird_mask.overlap(upperPipe_mask, upper_offset)
    l_collision = bird_mask.overlap(lowerPipe_mask, lower_offset)

    if u_collision or l_collision:
        return True
    return False
    
if __name__ == '__main__':
    while 1:
        base = Base()
        bird = Bird(WIN_HEIGHT//2-100)
        pipes = [Pipe()]
        clock = pygame.time.Clock()
        over = False
        while not over:
            render(win, bird, pipes, base)
            over = play_step(bird, pipes, base)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        bird.jump()
                if event.type == pygame.MOUSEBUTTONUP:
                    None
            
            clock.tick(30)