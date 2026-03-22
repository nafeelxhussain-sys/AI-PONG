import pygame
import random
import numpy as np

#game env
BLOCK = 10
SPEED = 60
MAX_SPEED = 20
OPP_SPEED = 0.7
MAX_OPP_SPEED = 0.7

class PingPong:
    def __init__(self, w=600, h=400):
        pygame.init()
        self.w = w 
        self.h = h
        self.paddle_length = BLOCK*5
        self.ball_radius = 7
        self.hit_right_this_step = False
        self.font = pygame.font.SysFont('Arial', 30)
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Ping Pong RL Environment")
        self.clock = pygame.time.Clock()
        self.reset()

    
    def reset(self):
        self.score1 = 0
        self.score2 = 0
        self.ball_x = random.randint(self.w//3, 2*self.w//3)
        self.ball_y = random.randint(0, self.h)
        self.ball_vx = random.choice([-3,3])
        self.ball_vy = random.uniform(-2,2)
        self.Lpaddle_y = self.h/2 - self.paddle_length/2
        self.Rpaddle_y = self.h/2 - self.paddle_length/2

        return self.get_state()

    def reset_rally(self):
        self.ball_x = random.randint(self.w//3, 2*self.w//3)
        self.ball_y = random.randint(self.h//3, 2*self.h//3)
        self.ball_vx = random.choice([-3,3])
        self.ball_vy = random.uniform(-2,2)
        self.Lpaddle_y = self.h/2 - self.paddle_length/2
        self.Rpaddle_y = self.h/2 - self.paddle_length/2

        return self.get_state()
    
    def render(self):
        self.display.fill((0,0,0))
    
        # Left paddle
        pygame.draw.rect(
            self.display,
            (255,255,255),
            pygame.Rect(10, self.Lpaddle_y , BLOCK, self.paddle_length)
        )
    
        # Right paddle
        pygame.draw.rect(
            self.display,
            (255,255,255),
            pygame.Rect(self.w-20, self.Rpaddle_y , BLOCK, self.paddle_length)
        )
    
        # Ball
        pygame.draw.circle(
            self.display,
            (255,255,255),
            (int(self.ball_x), int(self.ball_y)),
            self.ball_radius
        )

        # middle line
        pygame.draw.rect(
            self.display,
            (255,255,255),
            pygame.Rect(self.w/2, 0 , 1 , self.h)
        )

        # score
        score_txt = f"{self.score1}      {self.score2}"
        text_surface = self.font.render(score_txt, True, (255, 255, 255))
        self.display.blit(text_surface, (self.w/2 - 40, 20))

        self.clock.tick(SPEED)
    
        pygame.display.flip()

    def get_state(self):
        state = np.array([
            self.ball_x/self.w,
            self.ball_y/self.h,
            self.ball_vx/MAX_SPEED,
            self.ball_vy/MAX_SPEED,
            self.Rpaddle_y/(self.h-self.paddle_length),
            self.Lpaddle_y/(self.h-self.paddle_length)
            ],dtype=np.float32)
        
        return state.reshape(1,-1)




    def step(self, action):
        '''
            [1,0,0] --> move up
            [0,1,0] --> no move
            [0,0,1] --> move down
        '''
        self.hit_right_this_step = False
        # move left paddle
        # self.move_opponent()

        # take actioin on right paddle
        if action[0]==1:
            self.Rpaddle_y = max(0,self.Rpaddle_y - BLOCK)
        if action[2]==1:
            self.Rpaddle_y = min(self.h - self.paddle_length,self.Rpaddle_y + BLOCK)

        
        # move the ball one frame
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        self.collision()
        reward, done = self.get_info()
        next_state = self.get_state()

        return next_state, reward, done

    def collision(self):
        # top
        if self.ball_y - self.ball_radius <= 0 :
            self.ball_vy *= -1
            if abs(self.ball_vy) < 0.1:
                self.ball_vy == 0.1
        #bottom
        elif  self.ball_y + self.ball_radius >= self.h:
            self.ball_vy *= -1
            if abs(self.ball_vy) < 0.1:
                self.ball_vy == -0.1
            
    
        # LEFT PADDLE
        if self.ball_x - self.ball_radius <= 10 + BLOCK:
            if not (self.ball_y + self.ball_radius < self.Lpaddle_y or 
                    self.ball_y - self.ball_radius > self.Lpaddle_y + self.paddle_length):
                
                self.ball_x = 10 + BLOCK + self.ball_radius  # prevent sticking
                self.ball_vx = abs(self.ball_vx) * 1.1

                offset = (self.ball_y - (self.Lpaddle_y + self.paddle_length/2)) / (self.paddle_length/2)
                self.ball_vy += offset * 2

        # RIGHT PADDLE
        if self.ball_x + self.ball_radius >= self.w - 20:
            if not (self.ball_y + self.ball_radius < self.Rpaddle_y or 
                    self.ball_y - self.ball_radius > self.Rpaddle_y + self.paddle_length):

                self.hit_right_this_step = True
                self.ball_x = self.w - 20 - self.ball_radius  # prevent sticking
                self.ball_vx = -abs(self.ball_vx) * 1.1

                offset = (self.ball_y - (self.Rpaddle_y + self.paddle_length/2)) / (self.paddle_length/2)
                self.ball_vy += offset * 2
    
        self.ball_vx = max(-MAX_SPEED, min(MAX_SPEED, self.ball_vx))
        self.ball_vy = max(-MAX_SPEED, min(MAX_SPEED, self.ball_vy))


    def get_info(self):
        reward = 0.0
        done = False
    
        # left side miss (agent wins point)
        if self.ball_x - self.ball_radius< 20 and (self.Lpaddle_y > self.ball_y + self.ball_radius or 
            self.Lpaddle_y + self.paddle_length <  self.ball_y - self.ball_radius ):
            done = True
            reward += 1
            self.score2 += 1
    
        # right side miss (agent loses point)
        elif self.ball_x  + self.ball_radius> self.w - 30 and (self.Rpaddle_y > self.ball_y + self.ball_radius or 
            self.Rpaddle_y + self.paddle_length <  self.ball_y - self.ball_radius):
            done = True
            reward -= 1 + abs(self.Lpaddle_y - self.ball_y)
            self.score1 += 1
    
        # right paddle hit → give shaped reward
        elif self.hit_right_this_step:
            reward += 0.25

    
        return np.clip(reward,-10.0,10.0), done

    def move_opponent(self):

        if random.random() < 0.1:
            return
       
        paddle_center = self.Lpaddle_y + self.paddle_length / 2
        target = self.ball_y + random.uniform(-5, 5)
        diff = target - paddle_center


        if diff < -10:
            self.Lpaddle_y = max(0, self.Lpaddle_y - OPP_SPEED * BLOCK)
        elif diff > 10:
            self.Lpaddle_y = min(self.h - self.paddle_length, self.Lpaddle_y + OPP_SPEED * BLOCK)
            
    def move_agent(self,action):
        if action[0]==1:
            self.Rpaddle_y = max(0,self.Rpaddle_y - BLOCK)
        if action[2]==1:
            self.Rpaddle_y = min(self.h - self.paddle_length,self.Rpaddle_y + BLOCK)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
    def isFinished(self):
        if self.score1==5 or self.score2==5:
            return True
        return False

    def move_user(self):
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: # Move Up
            self.Lpaddle_y = max(0,self.Lpaddle_y - BLOCK)
        elif keys[pygame.K_DOWN]: # Move Down
            self.Lpaddle_y = min(self.h - self.paddle_length, self.Lpaddle_y + BLOCK)           