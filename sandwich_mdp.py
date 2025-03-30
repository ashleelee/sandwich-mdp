import numpy as np
import pygame
import time
import random
import gymnasium as gym
from gymnasium import spaces
from d_ingredients import dict_ingredients
from actions import dict_actions

# init variables for pygame
FPS = 30
VIEWPORT_W = 1330
VIEWPORT_H = 650

class Ingredient:
    def __init__(self, img_path, pos, size, visible=False, on_plate = False):
        self.img = pygame.transform.scale(pygame.image.load(img_path).convert_alpha(), size)
        self.pos = pos  # (x, y)
        self.size = size  # (width, height)
        self.visible = visible
        self.on_plate = on_plate
    
    def update_pos(self, x, y):
        self.pos = (x, y)
    
    def update_size(self, size_x, size_y):
        self.size = (size_x, size_y)  # update size tuple
        self.img = pygame.transform.scale(self.img, self.size)  # resize the image

    def draw(self, screen):
        # Draw the ingredient if it's visible
        if self.visible:
            screen.blit(self.img, self.pos)
    
    def draw_on_plate(self, screen, plate_y):
        top_left_x = 710 - self.size[0] // 2
        top_left_y = plate_y - self.size[1] // 2
        screen.blit(self.img, (top_left_x, top_left_y))

class Button:
    def __init__(self, pos, action, color=(255, 255, 255)):
        self.pos = pos
        self.color = color
        self.action = action
    

class SandwichMakingEnv(gym.Env):

    def __init__(self):
        """
        ### Action Space
        There are 38 discrete actions. Each action selects a different ingredient 
        or processes an ingredient in a specific way.
        
        ### Observation Space
        The state is represented by a 14-dimensional vector, 
        documenting the status of the 14 ingredients.

        ### Rewards
        A reward system is not set up for this environment, as it is designed for IL.

        ### Starting State
        The environment starts with every ingredient in the kitchen/storage.

        ### Episode Termination
        An episode is complete when the sandwich is finished, which occurs when the top bread is placed.
        """

        super().__init__()

        # init pygame variables
        self.screen: pygame.Surface = None
        self.clock = None
        self.load_ingredients()
        self.font = pygame.font.SysFont("calibri", size=16)
        self.font.set_bold(True)
        
        # list of tuples with dictionaries describing each component state
        self.dict_ingredients = dict_ingredients
        self.dict_actions = dict_actions
        self.action_space = spaces.Discrete(len(dict_actions))
        self.state = [0] * len(dict_ingredients) 
        self.performed_actions = set()
        self.performed_actions_list = []
        self.ingredients_on_plate = []
        self.help_buttons = []

        self.high = np.array([
            2,  # number of plate states
            3,  # number of bottom bread states
            3,  # number of regular mayo states
            3,  # number of veggie mayo states
            6,  # number of tomato states
            6,  # number of avocado states
            3,  # number of lettuce states
            6,  # number of eggs states
            3,  # number of ham states
            4,  # number of bacon states
            4,  # number of plant-based meat states
            3, # number of cheese states
            3,  # number of pepper states
            3   # number of top bread states
        ], dtype=np.int32) - 1 # subtracting 1 = the max value for each index

        # self.observation_space = spaces.Box(0, high, dtype=np.int32)
        self.observation_space = spaces.Box(0, self.high, shape=(len(self.high),), dtype=np.int32)
        self.done = False
        self.update_help_button()

    def load_ingredients(self):
        if not pygame.get_init():
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        
        # intialized with the initial position
        # starred position are the ones that would be on plate
        self.ingredients = {
            "shelf": Ingredient("img/shelf.png", (450, 90), (320, 320), True), # 1 pos
            "fridge": Ingredient("img/fridge.png", (10, 165), (260, 260), True), # 1 pos
            "pan": Ingredient("img/pan.png", (10, 400), (160, 160), True), # 1 pos
            "plate": Ingredient("img/plate.png", (670, 165), (100, 100), True), # 2 pos
            "bread": Ingredient("img/bread.png", (450, 125), (130, 130), True), # 1 pos
            "bread_bag": Ingredient("img/bread_bag.png", (450, 125), (130, 130), True), # 1 pos
            "b_bread": Ingredient("img/bottom_bread.png", (475, 450), (120, 120)), # 1 pos*
            "reg_mayo": Ingredient("img/regular_mayo.png", (530, 155), (100, 100), True), # 2 pos
            "reg_mayo_spr": Ingredient("img/regular_mayo_spread.png", (350, 400), (100, 100)), # 0 pos*
            "v_mayo": Ingredient("img/vegan_mayo.png", (570, 155), (100, 100), True), # 2 pos
            "v_mayo_spr": Ingredient("img/vegan_mayo_spread.png", (350, 400), (100, 100)), # 0 pos*
            "tomato": Ingredient("img/tomato.png", (85, 250), (65, 65), True), # 2 pos
            "s_tomato": Ingredient("img/sliced_tomato.png", (250, 450), (110, 110)), # 1 pos*
            "d_tomato": Ingredient("img/diced_tomato.png", (250, 450), (120, 120)), # 1 pos*
            "avocado": Ingredient("img/avocado.png", (27, 255), (70, 70), True),# 2 pos
            "s_avocado": Ingredient("img/sliced_avocado.png", (250, 540), (120, 120)), # 1 pos*
            "m_avocado": Ingredient("img/mashed_avocado.png", (245, 540), (120, 120)), # 1 pos*
            "lettuce": Ingredient("img/lettuce.png", (65, 197), (55, 55), True), # 2 pos*
            "eggs": Ingredient("img/eggs.png", (23, 190), (65, 65), True), # 2 pos
            "f_egg_1": Ingredient("img/one_fried_egg.png", (25, 430), (90, 90)), # 1 pos*
            "f_egg_2": Ingredient("img/two_fried_eggs.png", (25, 430), (90, 90)), # 1 pos*
            "ham": Ingredient("img/ham.png", (27, 320), (55, 55), True), # 2 pos*
            "r_bacon": Ingredient("img/raw_bacon.png", (60, 320), (55, 55), True), # 2 pos
            "f_bacon": Ingredient("img/fried_bacon.png", (25, 430), (90, 90)), # 1 pos*
            "r_pb_meat": Ingredient("img/raw_plant_based_meat.png", (90, 320), (55, 55), True), # 2 pos
            "f_pb_meat": Ingredient("img/fried_plant_based_meat.png", (25, 430), (90, 90)), # 1 pos*
            "cheese": Ingredient("img/cheese.png", (90, 200), (55, 55), True), # 2 pos*
            "pepper": Ingredient("img/pepper.png", (620, 183), (75, 75), True), # 2 pos
            "pepper_spr": Ingredient("img/sprinkled_pepper.png", (720, 70), (75, 75)), # 0 pos*
            "t_bread": Ingredient("img/top_bread.png", (500, 430), (120, 120)) # 1 pos*
        }
        
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(len(dict_ingredients), dtype=np.int32)
        self.performed_actions.clear()
        self.performed_actions_list = []
        self.ingredients_on_plate = []
        # reset the variables for the screen
        return self.state, {}

    def step(self, action):

        # check if the action is valid
        if not self.is_valid_action(action): 
            return self.state, 0, False, False, {"invalid":0}
        
        # add action into performed actions set
        self.performed_actions.add(action)
        self.performed_actions_list.append(action)
        
        # update the state after the action is performed
        self.update_state(action)

        # update help buttons
        self.update_help_button()
        
        # check if final state is reached
        done = self.is_done()

        # update screen variables
        self.update_ingredients(action)
        
        return self.state, 0, done, False, {} 



    def is_valid_action(self, action):
        
        # actions cannot be repeated
        if action in self.performed_actions: return False

        '''
        if we want to put ingredients (other than plate/bottom bread) on to the 
        sandwich, both the plate and bottom bread must already be placed
        '''
        if (action in [4, 6, 10, 11, 15, 16, 18, 22, 23, 25, 28, 31, 33, 35, 37]
            and (self.state[0] != 1 or self.state[1] != 2)): return False

        '''
        checking if ingredients are taken out/processed 
        before placing on the sandwich
        '''
        if action == 2 and (self.state[0] != 1 or self.state[1] != 1): return False
        elif action == 4 and self.state[2] != 1: return False
        elif action == 6 and self.state[3] != 1: return False
        elif action in [8, 9] and self.state[4] != 1: return False
        elif action == 10 and self.state[4] != 2: return False
        elif action == 11 and self.state[4] != 3: return False
        elif action in [13, 14] and self.state[5] != 1: return False
        elif action == 15 and self.state[5] != 2: return False
        elif action == 16 and self.state[5] != 3: return False
        elif action == 18 and self.state[6] != 1: return False
        elif action in [20, 21] and self.state[7] != 1: return False
        elif action == 22 and self.state[7] != 2: return False
        elif action == 23 and self.state[7] != 3: return False
        elif action == 25 and self.state[8] != 1: return False
        elif action == 27 and self.state[9] != 1: return False
        elif action == 28 and self.state[9] != 2: return False
        elif action == 30 and self.state[10] != 1: return False
        elif action == 31 and self.state[10] != 2: return False
        elif action == 33 and self.state[11] != 1: return False
        elif action == 35 and self.state[12] != 1: return False
        elif action == 37 and self.state[13] != 1: return False
        else:
            return True
        
    def update_state(self, action):
        # mapping of actions to corresponding state indices
        action_to_state_index = {
            0: 0,                               # plate
            1: 1,  2: 1,                        # bottom bread
            3: 2,  4: 2,                        # regular mayo
            5: 3,  6: 3,                        # veggie mayo
            7: 4,  8: 4,  9: 4, 10: 4, 11: 4,   # tomato
            12: 5, 13: 5, 14: 5, 15: 5, 16: 5,  # avocado
            17: 6, 18: 6,                       # lettuce
            19: 7, 20: 7, 21: 7, 22: 7, 23: 7,  # eggs
            24: 8, 25: 8,                       # ham
            26: 9, 27: 9, 28: 9,                # bacon
            29: 10, 30: 10, 31: 10,             # plant-based meat
            32: 11, 33: 11,                     # cheese
            34: 12, 35: 12,                     # pepper
            36: 13, 37: 13                      # top bread
        }
        
        # check if action is mapped to a state index (safety check)
        if action in action_to_state_index:
            state_index = action_to_state_index[action]
            
            # handle special cases (multiple ways to process tomato, avocado, and eggs)
            if action == 9: self.state[state_index] = 3
            elif action == 10: self.state[state_index] = 4
            elif action == 11: self.state[state_index] = 5
            elif action == 14: self.state[state_index] = 3
            elif action == 15: self.state[state_index] = 4
            elif action == 16: self.state[state_index] = 5
            elif action == 21: self.state[state_index] = 3
            elif action == 22: self.state[state_index] = 4
            elif action == 23: self.state[state_index] = 5
            else:
                # check we do not exceed max allowed value for the state
                self.state[state_index] = min(self.state[state_index] + 1, self.high[state_index])

    def update_ingredients(self, action):
        if action == 0: 
            self.ingredients["plate"].update_pos(630, 480)
            self.ingredients["plate"].update_size(160, 160)
        elif action == 1:
            self.ingredients["b_bread"].visible = True
            self.ingredients["bread"].visible = False
        elif action == 2:
            self.ingredients["b_bread"].on_plate = True
            self.ingredients_on_plate.append("b_bread")
        elif action == 3:
            self.ingredients["reg_mayo"].update_pos(430, 370)
        elif action == 4:
            self.ingredients["reg_mayo"].visible = False
            self.ingredients["reg_mayo_spr"].visible = True
            self.ingredients["reg_mayo_spr"].on_plate = True
            self.ingredients_on_plate.append("reg_mayo_spr")
        elif action == 5:
            self.ingredients["v_mayo"].update_pos(470, 370)
        elif action == 6:
            self.ingredients["v_mayo"].visible = False
            self.ingredients["v_mayo_spr"].visible = True
            self.ingredients["v_mayo_spr"].on_plate = True
            self.ingredients_on_plate.append("v_mayo_spr")
        elif action == 7: 
            self.ingredients["tomato"].update_pos(250, 450)
            self.ingredients["tomato"].update_size(100, 100)
        elif action == 8: 
            self.ingredients["tomato"].visible = False
            self.ingredients["s_tomato"].visible = True
        elif action == 9: 
            self.ingredients["tomato"].visible = False
            self.ingredients["d_tomato"].visible = True
        elif action == 10: 
            self.ingredients["s_tomato"].on_plate = True
            self.ingredients_on_plate.append("s_tomato")
        elif action == 11: 
            self.ingredients["d_tomato"].on_plate = True
            self.ingredients_on_plate.append("d_tomato")
        elif action == 12: 
            self.ingredients["avocado"].update_pos(250, 540)
            self.ingredients["avocado"].update_size(100, 100)
        elif action == 13: 
            self.ingredients["avocado"].visible = False
            self.ingredients["s_avocado"].visible = True
        elif action == 14: 
            self.ingredients["avocado"].visible = False
            self.ingredients["m_avocado"].visible = True
        elif action == 15: 
            self.ingredients["s_avocado"].on_plate = True
            self.ingredients_on_plate.append("s_avocado")
        elif action == 16: 
            self.ingredients["m_avocado"].on_plate = True
            self.ingredients_on_plate.append("m_avocado")
        elif action == 17: 
            self.ingredients["lettuce"].update_pos(360, 540)
            self.ingredients["lettuce"].update_size(120, 120)
        elif action == 18:
            self.ingredients["lettuce"].on_plate = True
            self.ingredients_on_plate.append("lettuce")
        elif action == 19: 
            self.ingredients["eggs"].update_pos(20, 540)
            self.ingredients["eggs"].update_size(110, 110)
        elif action == 20: 
            self.ingredients["eggs"].visible = False
            self.ingredients["f_egg_1"].visible = True
        elif action == 21: 
            self.ingredients["eggs"].visible = False
            self.ingredients["f_egg_2"].visible = True
        elif action == 22: 
            self.ingredients["f_egg_1"].on_plate = True
            self.ingredients_on_plate.append("f_egg_1")
        elif action == 23: 
            self.ingredients["f_egg_2"].on_plate = True
            self.ingredients_on_plate.append("f_egg_2")
        elif action == 24: 
            self.ingredients["ham"].update_pos(365, 450)
            self.ingredients["ham"].update_size(120, 120)
        elif action == 25:
            self.ingredients["ham"].on_plate = True
            self.ingredients_on_plate.append("ham")
        elif action == 26: 
            self.ingredients["r_bacon"].update_pos(150, 450)
            self.ingredients["r_bacon"].update_size(90, 90)
        elif action == 27:
            self.ingredients["r_bacon"].visible = False
            self.ingredients["f_bacon"].visible = True
        elif action == 28:
            self.ingredients["f_bacon"].on_plate = True
            self.ingredients_on_plate.append("f_bacon")
        elif action == 29: 
            self.ingredients["r_pb_meat"].update_pos(150, 550)
            self.ingredients["r_pb_meat"].update_size(90, 90)
        elif action == 30:
            self.ingredients["r_pb_meat"].visible = False
            self.ingredients["f_pb_meat"].visible = True
        elif action == 31:
            self.ingredients["f_pb_meat"].on_plate = True
            self.ingredients_on_plate.append("f_pb_meat")
        elif action == 32: 
            self.ingredients["cheese"].update_pos(475, 540)
            self.ingredients["cheese"].update_size(120, 120)
        elif action == 33:
            self.ingredients["cheese"].on_plate = True
            self.ingredients_on_plate.append("cheese")
        elif action == 34:
            self.ingredients["pepper"].update_pos(395, 395)
        elif action == 35:
            self.ingredients["pepper_spr"].on_plate = True
            self.ingredients_on_plate.append("pepper_spr")
        elif action == 36:
            self.ingredients["t_bread"].visible = True
        elif action == 37:
            self.ingredients["t_bread"].on_plate = True
            self.ingredients_on_plate.append("t_bread")
        else:
            pass

        
    def update_help_button(self):
        self.help_buttons = []
        x = 1088
        y = 119
        for i in range(len(dict_actions)):
            if(self.is_valid_action(i)):
                b = Button((x, y), i)
                y += 30
                self.help_buttons.append(b)

    def is_done(self):
        if self.state[-1] == 2: #top bread placed
            return True
        return False
    

    def get_help(self):
        # checks if the user hovers over or clicks a button
        mouse_pos = pygame.mouse.get_pos()
        selected_action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.MOUSEMOTION:
                # change button color when hovered
                for button in self.help_buttons:
                    bx, by = button.pos
                    if (bx - 7 <= mouse_pos[0] <= bx + 7) and (by - 7 <= mouse_pos[1] <= by + 7):
                        button.color = (255, 0, 0)  # red on hover
                    else:
                        button.color = (255, 255, 255)  # default white

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # detect button click
                for button in self.help_buttons:
                    bx, by = button.pos
                    if (bx - 7 <= mouse_pos[0] <= bx + 7) and (by - 7 <= mouse_pos[1] <= by + 7):
                        selected_action = button.action  # store selected action
                        return selected_action  # return action 

        return None  # no action selected yet
    
    def handle_interfere(self):
        start_time = pygame.time.get_ticks()  # get the start time
        self.interfere_button = pygame.Rect(1100, 350, 190, 60) 

        while True:
            elapsed_time = pygame.time.get_ticks() - start_time  # time elapsed

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if self.interfere_button.collidepoint(mouse_pos):
                        return True  # if clicked, go to help mode

            if elapsed_time > 5000:  # if 5 seconds pass, exit interfere mode
                return False

    def render(self, screen="help"):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
    
        # fill the screen with white bg
        self.screen.fill((255, 255, 255))

        # draw the two rect in bg
        pygame.draw.rect(self.screen, (167, 210, 221), (0, 355, 800, 90))
        pygame.draw.rect(self.screen, (255, 244, 223), (0, 435, 800, 215))

        # draw the history of action box
        pygame.draw.rect(self.screen, (255, 244, 223), (810, 0, 250, 650))
        history_text = self.font.render("History of Actions", True, (106, 62, 32))
        self.screen.blit(history_text, (860, 10))

        text_font = pygame.font.SysFont("calibri", size=14)
        action_count = 0;
        history_y = 30
        for action in self.performed_actions_list:
            action_count += 1
            text = text_font.render(f"{action_count}. {self.dict_actions[action]}", True, (106, 62, 32))
            self.screen.blit(text, (830, history_y))
            history_y += 17
        
        # draw the ingredients (not on plate)
        for ingredient in self.ingredients:
            if self.ingredients[ingredient].visible:
                if not self.ingredients[ingredient].on_plate:
                    self.ingredients[ingredient].draw(self.screen)
        
        # draw the ingredients (on plate) in order
        plate_y = 560
        for ingredient in self.ingredients_on_plate:
            self.ingredients[ingredient].draw_on_plate(self.screen, plate_y)
            plate_y -= 5

        # draw the user interaction portion
        robot_img = pygame.transform.scale(pygame.image.load("img/robot.png").convert_alpha(), (75, 75))
        speech_img = pygame.transform.scale(pygame.image.load("img/speech_bubble.png").convert_alpha(), (150, 150))
        self.screen.blit(robot_img, (1090, 10))
        self.screen.blit(speech_img, (1135, -20))
        pygame.draw.rect(self.screen, (242, 204, 141), (1070, 90, 250, 600))
        
        if(screen == "help"):
            help_font = pygame.font.SysFont("calibri", size=16)
            help_font.set_bold(True)
            help_text_1 = help_font.render("I need", True, (106, 62, 32))
            help_text_2 = help_font.render("HELP!", True, (106, 62, 32))
            self.screen.blit(help_text_1, (1193, 33))
            self.screen.blit(help_text_2, (1193, 53))

            # print out valid action options
            options_font = pygame.font.SysFont("calibri", size=16)
            option_y = 110
            for button in self.help_buttons:
                options_text = options_font.render(self.dict_actions[button.action], True, (0, 0, 0))
                self.screen.blit(options_text, (1100, option_y))
                pygame.draw.circle(self.screen, button.color, button.pos, 7, 0)
                option_y += 30
        elif(screen == "interfere"):
            pygame.draw.rect(self.screen, (255, 244, 223), (1100, 350, 190, 60))
            interfere_font = pygame.font.SysFont("calibri", size=20)
            interfere_text = interfere_font.render("Interfere", True, (106, 62, 32))
            self.screen.blit(interfere_text, (1160, 370))

            progress_font = pygame.font.SysFont("calibri", size=16)
            progress_text_1 = progress_font.render("Do you want", True, (106, 62, 32))
            progress_text_2 = progress_font.render("to interfere?", True, (106, 62, 32))
            self.screen.blit(progress_text_1, (1177, 35))
            self.screen.blit(progress_text_2, (1180, 55))
        else:
            progress_font = pygame.font.SysFont("calibri", size=16)
            progress_text_1 = progress_font.render("Processing", True, (106, 62, 32))
            progress_text_2 = progress_font.render("action...", True, (106, 62, 32))
            self.screen.blit(progress_text_1, (1185, 35))
            self.screen.blit(progress_text_2, (1193, 55))
            
        self.clock.tick(FPS)
        pygame.display.flip()

def controlled_delay(delay_time):
    # non-blocking delay while keeping pygame responsive
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < delay_time:
        pygame.event.pump()  # keep event queue active to prevent freezing

if __name__ == "__main__":
    env = SandwichMakingEnv()
    obs, _ = env.reset()
    done = False
    screen_state = "process_action"
    info = {}
    need_new_action = True

    try:
        while not done:
            if need_new_action:
                action = env.action_space.sample()
                while(env.is_valid_action(action) == False):
                    env.render(screen_state)
                    action = env.action_space.sample()
                needs_help = random.random() < 0.50

                if needs_help:
                    screen_state = "help"
                else:
                    screen_state = "interfere"
                need_new_action = False
            
            if screen_state == "interfere":
                env.render("interfere")
                interfere_status = env.handle_interfere() 
                if interfere_status == True:
                    screen_state = "help"
                else:
                    screen_state = "process_action"

            if screen_state == "help":
                env.render(screen_state)
                action = env.get_help()
                if action != None:
                    screen_state = "process_action"

            if screen_state == "process_action":
                obs, _, done, _, info = env.step(action)  # originally state, rewards, done, truncated, info
                # save the state (before & after), action itself, context
                # context = ["robot independent", "robot asked", "human intervened"]
                # if == robot asked/human intervened, save robot prediction & human correction
                if info == {}:
                    
                    env.render(screen_state)
                    controlled_delay(2000)  # Wait for 1 second between each step
                    
                need_new_action = True

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        controlled_delay(3000)  # give time to see final state before closing
        env.close()  # Close the Gym environment
        del env  # Explicitly delete the environment

        # Ensure pygame quits properly
        if pygame.get_init():
            pygame.quit()

        print("Environment closed successfully.")


# restart - input(num of episode), opening window of, on episode ___
# extend user interfering 5 seconds, counting 
# output could be invalid
# logging - state of env of every episode, aciton proposed by the robot, human interfere - , what action gets executed, 
    # who executed
# data logger set - pytorch data set, --> each episode (is an object)
# class - replay buffer, episode, (all the episode)
# class - replay (for whole data set), episode ()

# initial script: synthetic data generation script
# function taking in user preferences (e.g. egg sandwich) -> generate possible sequences
    # key word bank - egg, ham, tomato, veggie, plant-based meat, bacon -> k arbitrary sequences (sample episode, same data structure)
    # change the sample function, constrain such as plate - then bread - action always be egg afterwards -> always put the ingredient onto it after it's taken out -> 20-25
# set of user preferences, some pr
