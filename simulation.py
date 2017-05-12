#!/usr/bin/env python4

import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
import pygame_util
import pymunk.util
from pymunk import Vec2d
from math import *
import random

from walker import walker

class Simulation():
        def __init__(self, scr_w = 600, scr_h = 600, \
                        angle=pi/20, gravity=200, \
                        show=True):
                # Window size
                self.scr_w = scr_w
                self.scr_h = scr_h
                # Graphics initialization (if show)
                self.show = show
                if self.show:
                        pygame.init()
                        self.screen = \
                          pygame.display.set_mode((self.scr_w,
                                  self.scr_h))
                        self.clock = pygame.time.Clock()
                # Create the space
                self.space = pymunk.Space()
                self.space.gravity = (0, -gravity)
                # Create the floor 
                self._create_floor(angle)
        def _create_floor(self, angle):
                # Create the floor
                body = pymunk.Body()
                body.position = self.scr_w/2, self.scr_h/4
                v = [(-self.scr_w,-(self.scr_h/2)*sin(angle)), \
                     (self.scr_w,(self.scr_h/2)*sin(angle)), \
                     (self.scr_w, -self.scr_h/2), \
                     (-self.scr_w, -self.scr_h/2)]
                floor = pymunk.Poly(body, v)
                floor.friction = 1.0
                floor.elasticity = 0.4
                self.space.add(floor)
        def _invy(self, pos):
                return pos[0], self.scr_h - pos[1]
        def step(self, delta):
                # Simulation step
                self.space.step(delta)
                # Draw stuff (if show)
                if self.show:
                        self.screen.fill(THECOLORS['black'])
                        pygame_util.draw_space(self.screen, self.space)
                        pygame.display.flip()
                        self.clock.tick(1/delta)
        def interactive(self):
                # Interactive mode
                running = True
                robot = None
                while running:
                        # Deal with clicks and other events
                        for event in pygame.event.get():
                                if event.type == QUIT:
                                        running = False
                                if event.type == MOUSEBUTTONDOWN:
                                        robot = walker(self.space,
                                                self._invy(event.pos),
                                                80, 60, 10, pi/16, 0, 0, 0)
                        self.step(0.02)

        def put_robot(self, robot):
                self.robot = robot

        def get_ke(self):
                # Calculates the kinect energy of the simulation
                k = 0.0
                for body in self.space.bodies:
                        k += body.kinetic_energy
                return k

        def mySimul(self, pos, ul, ll, w, lua, lla, rua, rla):

            robot = walker(self.space,
                           pos, ul, ll, w, lua, lla, rua, rla)

            iterations = 0
            while True:
                self.step(0.02)
                iterations += 1

                if self.get_ke() < 20:
                    return iterations, self.space.bodies[1].position


if __name__ == '__main__':
    s = Simulation()
    s.mySimul()

