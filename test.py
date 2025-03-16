self.dict_ingredients = [
            {0: "plate not placed", 1: "plate placed"}, # any buffer zone needed?
            {0: "bottom bread in bags", 1: "bottom bread taken out", 2: "bottom bread placed"},
            {0: "no dressing selected", 1: "regular mayo selected", 2: "spicy mayo selected", 
             3: "veggie mayo selected", 4: "regular mayo spread", 5: "spicy mayo spread", 
             6: "veggie mayo spread"},
            {0: "tomato in fridge", 1: "tomato taken out of fridge", 
             2: "tomato sliced", 3: "tomato diced", 4: "tomato placed"},
            {0: "avocado in fridge", 1: "avocado taken out of fridge", 
             2: "avocado sliced", 3: "avocado chopped", 4: "avocado mashed", 
             5: "avocado placed"},
            {0: "lettuce in fridge", 1: "lettuce taken out of fridge", 
             2: "lettuce placed"},
            {0: "eggs in fridge", 1: "eggs taken out of fridge", 
             2: "1 egg in the pan", 3: "2 eggs in the pan", 
             4: "1 boiled egg placed", 5: "2 boiled eggs placed"},
            {0: "ham in fridge", 1: "ham taken out of fridge", 
             2: "1 ham selected", 3: "2 ham selected", 
             4: "1 ham placed", 5: "2 ham placed"},
            {0: "meat in fridge", 1: "bacon taken out of fridge", 
             2: "plant-based meat taken out of fridge", 3: "bacon fried", 
             4: "plant-based meat fried", 5: "bacon placed", 
             6: "plant-based meat placed"}, 
            {0: "chesse still in fridge", 1: "normal cheese taken out of fridge", 
             2: "plant-based cheese taken out of fridge", 3: "1 normal cheese placed", 
             4: "1 plant-based cheese placed", 5: "2 normal cheese placed", 
             6: "2 plant-based cheese placed"}
            {0: "pepper still in storage", 1: "pepper taken out", 
             2: "pepper sprinkled"},
            {0: "top bread not placed", 1: "top bread taken", 2: "top bread placed"},
        ]
        """
        will this account for order?
        bottom bread on top of tomato vs tomato on top of bottom bread 
         - would be the same in this vector
        """

        """
        does the # of each ingredient state matter? 
        meat -> separate into ham/veggie/bacon or just meat?
        """

# on_plate tested first! in if statement?

dict_actions = {
    # plate
    0: "Take a plate", # plate pos + size changed

    # bottom bread
    1: "Take bottom bread out of bag", # bottom bread visible on, bread visible off
    2: "Place bottom bread on the plate", # bottom bread on_plate true

    # regular mayo
    3: "Take regular mayo dressing", # mayo pos + size changed
    4: "Spread regular mayo dressing", # mayo spread visible on + on_plate true, mayo visible off

    # veggie mayo
    5: "Take veggie mayo dressing", # v mayo pos + size changed
    6: "Spread veggie mayo dressing", # v mayo spread visible on + on_plate true,v mayo visible off

    # tomato
    7: "Take tomato out of fridge", # tomato pos + size changed
    8: "Slice tomato", # s tomato visible on, tomato visible off
    9: "Dice tomato", # d tomato visible on, tomato visible off
    10: "Place sliced tomato", # s tomato on_plate true
    11: "Place diced tomato", # d tomato on_plate true

    # Avocado processing
    12: "Take avocado out of fridge", # avocado pos + size changed
    13: "Slice avocado", # s avocado visible on, avocado visible off
    14: "Mash avocado", # m avocado visible on, avocado visible off
    15: "Place sliced avocado", # s avocado on_plate true
    16: "Place mashed avocado", # m avocado on_plate true

    # Lettuce
    17: "Take lettuce out of fridge", # lettuce pos + size changed
    18: "Place lettuce", # lettuce on_plate true

    # Eggs processing
    19: "Take eggs out of fridge", # egg pos + size changed
    20: "Crack 1 egg into the pan", # f 1 egg visible on, eggs visible off
    21: "Crack 2 eggs into the pan", # f 2 eggs visible on, eggs visible off
    22: "Place 1 fried egg", # f 1 egg on_plate true
    23: "Place 2 fried eggs", # f 2 eggs on_plate true

    # Ham processing
    24: "Take ham out of fridge", # ham pos + size changed
    25: "Place ham", # ham on_plate true

    # Meat processing
    26: "Take bacon out of fridge", # r_bacon pos + size changed
    27: "Fry bacon", # f_bacon visible on, r_bacon visible off
    28: "Place bacon", # r_bacon on_plate true

    29: "Take plant-based meat out of fridge", # r_pb_meat pos + size changed
    30: "Fry plant-based meat", # f pb meat visible on, r pb meat visible off
    31: "Place plant-based meat", # f pb meat on_plate true

    # Cheese processing
    32: "Take cheese out of fridge", # cheese pos + size chagned
    33: "Place cheese", # cheese_on_plate true

    # Pepper
    34: "Take pepper out of storage", # pepper pos + size changed
    35: "Sprinkle pepper", # pepper_spr visible on + on_plate true, pepper visible off

    # Top bread
    36: "Take top bread out of bag", # top bread visible on
    37: "Place top bread" # top bread on_plate true
}


self.ingredients = {
    "shelf": Ingredient("img/shelf.png", (550, -10), (320, 320), True), # 1 pos
    "fridge": Ingredient("img/fridge.png", (10, -5), (260, 260), True), # 1 pos
    "pan": Ingredient("img/pan.png", (10, 170), (160, 160), True), # 1 pos
    "plate": Ingredient("img/plate.png", (770, 65), (100, 100), True), # 2 pos
    "bread": Ingredient("img/bread.png", (550, 25), (130, 130), True), # 1 pos
    "bread_bag": Ingredient("img/bread_bag.png", (550, 25), (130, 130), True), # 1 pos
    "b_bread": Ingredient("img/bottom_bread.png", (465, 220), (120, 120), on_plate = True), # 1 pos*
    "reg_mayo": Ingredient("img/regular_mayo.png", (630, 55), (100, 100), True), # 2 pos
    "reg_mayo_spr": Ingredient("img/regular_mayo_spread.png", (350, 400), (100, 100), on_plate = True), # 0 pos*
    "v_mayo": Ingredient("img/vegan_mayo.png", (670, 55), (100, 100), True), # 2 pos
    "v_mayo_spr": Ingredient("img/vegan_mayo_spread.png", (350, 400), (100, 100), on_plate = True), # 0 pos*
    "tomato": Ingredient("img/tomato.png", (85, 80), (65, 65), True), # 2 pos
    "s_tomato": Ingredient("img/sliced_tomato.png", (250, 220), (120, 120), on_plate = True), # 1 pos*
    "d_tomato": Ingredient("img/diced_tomato.png", (250, 220), (120, 120), on_plate = True), # 1 pos*
    "avocado": Ingredient("img/avocado.png", (27, 85), (70, 70), True),# 2 pos
    "s_avocado": Ingredient("img/sliced_avocado.png", (250, 310), (120, 120), on_plate = True), # 1 pos*
    "m_avocado": Ingredient("img/mashed_avocado.png", (250, 310), (120, 120), on_plate = True), # 1 pos*
    "lettuce": Ingredient("img/lettuce.png", (65, 27), (55, 55), True, True), # 2 pos*
    "eggs": Ingredient("img/eggs.png", (23, 20), (65, 65), True), # 2 pos
    "f_egg_1": Ingredient("img/one_fried_egg.png", (25, 190), (90, 90), on_plate = True), # 1 pos*
    "f_egg_2": Ingredient("img/two_fried_eggs.png", (25, 200), (90, 90), on_plate = True), # 1 pos*
    "ham": Ingredient("img/ham.png", (27, 150), (55, 55), True, True), # 2 pos*
    "r_bacon": Ingredient("img/raw_bacon.png", (60, 150), (55, 55), True), # 2 pos
    "f_bacon": Ingredient("img/fried_bacon.png", (25, 200), (90, 90), on_plate = True), # 1 pos*
    "r_pb_meat": Ingredient("img/raw_plant_based_meat.png", (90, 150), (55, 55), True), # 2 pos
    "f_pb_meat": Ingredient("img/fried_plant_based_meat.png", (25, 200), (90, 90), on_plate = True), # 1 pos*
    "cheese": Ingredient("img/cheese.png", (90, 30), (55, 55), True, True), # 2 pos*
    "pepper": Ingredient("img/pepper.png", (720, 83), (75, 75), True), # 2 pos
    "pepper_spr": Ingredient("img/sprinkled_pepper.png", (720, 70), (75, 75), on_plate = True), # 0 pos*
    "t_bread": Ingredient("img/top_bread.png", (500, 200), (120, 120), on_plate = True) # 1 pos*
}