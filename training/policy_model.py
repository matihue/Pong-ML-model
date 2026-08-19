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

        self.ball = Ball(x=100, y=500, radius=20, game=self)

        self.rects = [
            Rect(x=50, y=100, width=50, height=300, game=self),
            Rect(x=1400, y=100, width=50, height=300, game=self),
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

        self.target_state_left = (
            None  ### state of ball and LEFT palette,  RIGHT attacking
        )
        self.target_state_right = (
            None  ### state of ball and RIGHT palette, LEFT attacking
        )

        self.now_attacking = None  ### 0 - left palette, 1 - right palette

        ### saving both defensive and attacking decisions to calculate error ###
        self.defensive_action_left = None
        self.defensive_action_right = None

        self.attacking_action_left = None
        self.attacking_action_right = None

        ### attack errors both boolean - indicating impact of attack decision on overall error ###
        self.attack_error_left = None
        self.attack_error_right = None

        ### how many iterations it will take for the ball to complete distance between palettes ###
        self.iterations = 0

        self.move_left = None
        self.move_right = None

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

    ### ### ### ###

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

        ### move ###
        if self.move_left:
            self.rects[0].y += self.move_left / self.iterations
        if self.move_right:
            self.rects[1].y += self.move_right / self.iterations
        self.rects[1].y = np.clip(self.rects[1].y, 0, 600)
        self.rects[0].y = np.clip(self.rects[0].y, 0, 600)

        ### wall collision ###
        if self.ball.rect.colliderect(
            self.upper_rects[0].rect
        ) or self.ball.rect.colliderect(self.upper_rects[1].rect):
            self.ball.y_dir *= -1

        ### indicate error for later learning ###
        if self.ball.x <= 120 and self.now_attacking is not None:
            target = self.ball.y

            base_pos = self.defensive_action_left * 900
            with_off = base_pos + 300 * self.attacking_action_left

            base_error = target - base_pos
            off_error = target - with_off

            self.attack_error_left = abs(off_error) > abs(base_error)

        elif self.ball.x >= 1380 and self.now_attacking is not None:
            target = self.ball.y

            base_pos = self.defensive_action_right * 900
            with_off = base_pos + 300 * self.attacking_action_right

            base_error = target - base_pos
            off_error = target - with_off

            self.attack_error_right = abs(off_error) > abs(base_error)

        ### reset if game over ###
        if self.ball.rect.colliderect(
            self.side_rects[0].rect
        ) or self.ball.rect.colliderect(self.side_rects[1].rect):
            if self.now_attacking == 0:
                PongModel.train_offensive(
                    self.target_state_left, self.attacking_action_left, 1
                )
                if self.attack_error_right:
                    PongModel.train_offensive(
                        self.target_state_right, self.attacking_action_right, -1
                    )

                self.now_attacking = None

            elif self.now_attacking == 1:
                PongModel.train_offensive(
                    self.target_state_right, self.attacking_action_right, 1
                )
                if self.attack_error_left:
                    PongModel.train_offensive(
                        self.target_state_left, self.attacking_action_left, -1
                    )

                self.now_attacking = None

            self.ball.game.state = "start"
            self.score = 0
            self.ball.x = 100
            self.ball.y = 450
            self.ball.direction = 1
            self.ball.angle = 0
            self.ball.y_dir = 1
            self.rects[0].y = self.rects[1].y = 300

            predictor_state = (self.ball.y / 900, self.ball.angle, self.ball.y_dir)

            self.target_state_right = (
                self.rects[1].y / 600,
                self.ball.v,
                self.ball.angle,
                self.ball.y_dir,
                self.ball.y / 900,
            )

            prediction = PongModel.predict(predictor_state)
            offensive_prediction = PongModel.predict_offensive(self.target_state_right)
            final_prediction = (prediction * 900 - 150) + offensive_prediction[1] * 300

            self.now_attacking = 0

            self.defensive_action_right = prediction
            self.attacking_action_right = offensive_prediction[1]

            self.move_right = final_prediction - self.rects[1].y
            self.iterations = self.calc_iters()

        ### right palette collision ###
        if self.ball.direction == 1 and self.ball.rect.colliderect(self.rects[1].rect):
            self.ball.direction = -1
            self.ball.angle = self.find_angle(self.rects[1])
            self.score += 1

            predictor_state = (self.ball.y / 900, self.ball.angle, self.ball.y_dir)

            self.target_state_left = (
                self.rects[0].y / 600,
                self.ball.v,
                self.ball.angle,
                self.ball.y_dir,
                self.ball.y / 900,
            )

            prediction = PongModel.predict(predictor_state)
            offensive_prediction = PongModel.predict_offensive(self.target_state_left)
            final_prediction = (prediction * 900 - 150) + offensive_prediction[1] * 300

            self.now_attacking = 1

            self.defensive_action_left = prediction
            self.attacking_action_left = offensive_prediction[1]

            self.move_left = final_prediction - self.rects[0].y
            self.iterations = self.calc_iters()

        ### left palette collision ###
        elif self.ball.direction == -1 and self.ball.rect.colliderect(
            self.rects[0].rect
        ):
            self.ball.direction = 1
            self.ball.angle = self.find_angle(self.rects[0])
            self.score += 1

            predictor_state = (self.ball.y / 900, self.ball.angle, self.ball.y_dir)
            self.target_state_right = (
                self.rects[1].y / 600,
                self.ball.v,
                self.ball.angle,
                self.ball.y_dir,
                self.ball.y / 900,
            )

            prediction = PongModel.predict(predictor_state)
            offensive_prediction = PongModel.predict_offensive(self.target_state_right)
            final_prediction = (prediction * 900 - 150) + offensive_prediction[1] * 300

            self.now_attacking = 0

            self.defensive_action_right = prediction
            self.attacking_action_right = offensive_prediction[1]

            self.move_right = final_prediction - self.rects[1].y
            self.iterations = self.calc_iters()

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
        self.v = 8

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
            PongModel.save()
            game.running = False

    if game.state == "start":
        game.screen.fill((0, 0, 0))

        game.update()
        game.draw()

    pygame.display.flip()
    game.clock.tick(600)

pygame.quit()
