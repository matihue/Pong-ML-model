import math
import random

import numpy as np
import pygame

from ..model import Model

PongModel = Model()

pygame.init()


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((1500, 900))
        self.clock = pygame.time.Clock()

        self.running = True
        self.score = 0

        self.ball = Ball(x=100, y=450, radius=20, game=self)

        self.rects = [
            Rect(x=50, y=300, width=50, height=300, game=self),
            Rect(x=1400, y=300, width=50, height=300, game=self),
        ]

        self.upper_rects = [
            Rect(x=0, y=0, width=1500, height=1, game=self),
            Rect(x=0, y=900, width=1500, height=1, game=self),
        ]

        self.side_rects = [
            Rect(x=-20, y=-900, width=2, height=1900, game=self),
            Rect(x=1520, y=-900, width=2, height=1900, game=self),
        ]

        self.state = "start"
        self.objects = [self.ball, self.rects[0], self.rects[1]]

        self.pending_state_left = None
        self.pending_state_right = None

    ### helpers ###
    def find_angle(self):
        return random.uniform(
            -self.ball.max_angle, self.ball.max_angle
        )  ### random angle to generate more unusual data

    ### ### ### ###

    ### update game ###
    def update(self):

        ### move ball ####
        dx = math.cos(self.ball.angle)
        dy = math.sin(self.ball.angle)

        self.ball.x += self.ball.direction * dx * self.ball.v
        self.ball.y -= self.ball.y_dir * dy * self.ball.v

        self.ball.update_rect()
        self.rects[0].update_rect()
        self.rects[1].update_rect()

        ### wall collision ###
        if self.ball.rect.colliderect(
            self.upper_rects[0].rect
        ) or self.ball.rect.colliderect(self.upper_rects[1].rect):
            self.ball.y_dir *= -1

        ### teleport to ball (right palette) ###
        if self.ball.x >= 1380 and self.pending_state_right is not None:
            target = self.ball.y / 900
            self.rects[1].y = self.ball.y - 150
            self.rects[1].y = np.clip(self.rects[1].y, 0, 600)
            PongModel.save_train_data(self.pending_state_right, target)
            self.pending_state_right = None

        ### teleport to ball (left palette) ###
        elif self.ball.x <= 120 and self.pending_state_left is not None:
            target = self.ball.y / 900

            self.rects[0].y = self.ball.y - 150
            self.rects[0].y = np.clip(self.rects[0].y, 0, 600)
            PongModel.save_train_data(self.pending_state_left, target)
            self.pending_state_left = None

        ### right palette collision ###
        if self.ball.direction == 1 and self.ball.rect.colliderect(self.rects[1].rect):
            self.ball.direction = -1
            self.ball.angle = self.find_angle()
            self.score += 1

            self.pending_state_left = (
                self.ball.y / 900,
                self.ball.angle,
                self.ball.y_dir,
            )

        ### left palette collision ###
        elif self.ball.direction == -1 and self.ball.rect.colliderect(
            self.rects[0].rect
        ):
            self.ball.direction = 1
            self.ball.angle = self.find_angle()
            self.score += 1

            self.pending_state_right = (
                self.ball.y / 900,
                self.ball.angle,
                self.ball.y_dir,
            )

    ### ### ### ### ###

    ### visual ###
    def draw(self):
        for obj in self.objects:
            obj.draw()

        font = pygame.font.SysFont(None, 170)
        text_surface = font.render(f"{self.score}", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(750, 100))
        self.screen.blit(text_surface, text_rect)

    ### ### ### ##


class Ball:
    def __init__(self, x, y, radius, game):
        self.game = game

        self.max_angle = math.pi / 4
        self.angle = 0
        self.v = 6

        self.x = x
        self.y = y
        self.r = radius

        self.direction = 1
        self.y_dir = 1

        self.rect = pygame.Rect(
            self.x - self.r, self.y - self.r, self.r * 2, self.r * 2
        )

    def update_rect(self):
        self.rect = pygame.Rect(
            self.x - self.r, self.y - self.r, self.r * 2, self.r * 2
        )

    def draw(self):
        pygame.draw.circle(self.game.screen, (255, 255, 255), (self.x, self.y), self.r)


class Rect:
    def __init__(self, x, y, width, height, game):
        self.game = game

        self.x = x
        self.y = y

        self.width = width
        self.height = height

        self.center = self.y + self.height / 2

        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update_rect(self):
        self.center = self.y + self.height / 2
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self):
        pygame.draw.rect(
            self.game.screen, (255, 255, 255), (self.x, self.y, self.width, self.height)
        )


game = Game()
while game.running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.running = False

    if game.state == "start":
        game.screen.fill((0, 0, 0))

        game.update()
        game.draw()

    pygame.display.flip()
    game.clock.tick(600)

pygame.quit()
