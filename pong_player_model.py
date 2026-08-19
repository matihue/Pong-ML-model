import math

import numpy as np
import pygame

from model import Model

PongModel = Model()

pygame.init()


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((1500, 900))
        self.clock = pygame.time.Clock()

        self.running = True
        self.score = 0
        self.v_multiplier = 1.03

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

        self.model_move = None
        self.iterations = 0

    ### helpers ###
    def calc_iters(self):
        dx = 1280
        v = self.ball.v
        angle = self.ball.angle
        vx = v * math.cos(angle)

        return dx / vx

    def find_angle(self, touched):
        diff = 2 * (self.ball.y - touched.center)

        v = diff / touched.height

        return v * self.ball.max_angle

    ### ### #### ###

    ### update game ###
    def update(self):

        ### move ball ###
        dx = math.cos(self.ball.angle)
        dy = math.sin(self.ball.angle)

        self.ball.x += self.ball.direction * dx * self.ball.v
        self.ball.y -= self.ball.y_dir * dy * self.ball.v

        self.ball.update_rect()
        self.rects[0].update_rect()
        self.rects[1].update_rect()

        ### move (model) ###
        if self.model_move:
            self.rects[1].y += np.clip(self.model_move / self.iterations, -1, 1)
        self.rects[1].y = np.clip(self.rects[1].y, 0, 600)

        ### move (player) ###
        self.handle_moving()

        ### wall collision ###
        if self.ball.rect.colliderect(
            self.upper_rects[0].rect
        ) or self.ball.rect.colliderect(self.upper_rects[1].rect):
            self.ball.y_dir *= -1

        ### reset if game over ###
        if self.ball.rect.colliderect(
            self.side_rects[0].rect
        ) or self.ball.rect.colliderect(self.side_rects[1].rect):
            self.ball.game.state = "start"
            self.score = 0
            self.ball.x = 100
            self.ball.y = 450
            self.ball.direction = 1
            self.ball.angle = 0
            self.ball.y_dir = 1
            self.ball.v = 1.02
            self.rects[0].y = self.rects[1].y = 300
            self.model_move = None

        ### right palette collision (player) ###
        if self.ball.direction == 1 and self.ball.rect.colliderect(self.rects[1].rect):
            self.ball.direction = -1
            self.ball.angle = self.find_angle(self.rects[1])
            self.ball.v *= self.v_multiplier
            self.score += 1

            self.model_move = None

        ### left palette collision (model) ###
        elif self.ball.direction == -1 and self.ball.rect.colliderect(
            self.rects[0].rect
        ):
            self.ball.direction = 1
            self.ball.angle = self.find_angle(self.rects[0])
            self.ball.v *= self.v_multiplier
            self.score += 1

            predictor_state = (self.ball.y / 900, self.ball.angle, self.ball.y_dir)
            offensive_state = (
                self.rects[0].y / 600,
                self.ball.v,
                self.ball.angle,
                self.ball.y_dir,
                self.ball.y / 900,
            )
            prediction = PongModel.predict(predictor_state)
            offensive_prediction = PongModel.predict_offensive(offensive_state)
            final_prediction = (prediction * 900 - 150) + offensive_prediction[1] * 250

            self.model_move = final_prediction - self.rects[1].y

            self.iterations = self.calc_iters()

    ### ### ### ### ###

    ### player palette moving ###
    def handle_moving(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_w] and self.rects[0].y >= 0:
            self.rects[0].y -= 1

        if keys[pygame.K_s] and self.rects[0].y <= 600:
            self.rects[0].y += 1

    ### ### ### ### ### ### ### #

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
        self.v = 1.02

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
