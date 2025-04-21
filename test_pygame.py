import pygame
import sys

# Initialize pygame
pygame.init()

# Set up the display
WINDOW_SIZE = (700, 700)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("700x700 Jam Spreading Env Preview")

# Set background color
background_color = (255, 255, 255)  # Light gray

# Load bread image
bread_img = pygame.image.load("img_c/bread.png").convert_alpha()

# Optionally scale the image to fit a certain size
bread_img = pygame.transform.scale(bread_img, (550, 550))  # Adjust as needed

# Set position for the bread image (centered)
bread_pos = ((WINDOW_SIZE[0] - bread_img.get_width()) // 2,
             (WINDOW_SIZE[1] - bread_img.get_height())-30)

# Main loop
running = True
while running:
    screen.fill(background_color)
    
    # Draw the bread image
    screen.blit(bread_img, bread_pos)

    # Define colors
    box_color = (255, 191, 64)   # Orange-yellow
    text_color = (0, 0, 0)

    # Define box size
    box_size = 80

    # Define box positions
    start_pos = (50, 50)               # x, y for top-left corner
    bag_init_pos = (WINDOW_SIZE[0] - 50 - box_size, 50)

    # Draw the start position box
    pygame.draw.rect(screen, box_color, (*start_pos, box_size, box_size), width=5)

    # Draw the bag initial position box
    pygame.draw.rect(screen, box_color, (*bag_init_pos, box_size, box_size), width=5)

    # Inside your main loop, after screen.fill(...)
    mouse_x, mouse_y = pygame.mouse.get_pos()

    # Draw crosshair lines
    pygame.draw.line(screen, (150, 150, 150), (mouse_x, 0), (mouse_x, WINDOW_SIZE[1]), 1)
    pygame.draw.line(screen, (150, 150, 150), (0, mouse_y), (WINDOW_SIZE[0], mouse_y), 1)

    # Draw coordinates text
    font = pygame.font.SysFont(None, 24)
    coord_text = font.render(f"({mouse_x}, {mouse_y})", True, (0, 0, 0))
    screen.blit(coord_text, (mouse_x + 10, mouse_y + 10))

    box_positions = [
        (160, 155),
        (160, 275),
        (160, 395),
        (160, 515),
        (350, 155),
        (350, 275),
        (350, 395),
        (350, 515)
    ]

    box_width = 190
    box_height = 120
    box_color = (255, 100, 100)  # Light red for visibility
    border_width = 2  # Thickness of the rectangle edge

    for pos in box_positions:
        pygame.draw.rect(screen, box_color, (*pos, box_width, box_height), width=border_width)

    # Draw a thick line from (213, 243) to (252, 243)
    line_color = (200, 0, 0)  # Red color
    start_point = (213, 243)
    end_point = (252, 243)
    line_width = 30
    # pygame.draw.line(screen, line_color, start_point, end_point, line_width)

    # Draw a second thick line from (252, 243) to (276, 242)
    second_start = (252, 243)
    second_end = (276, 240)
    pygame.draw.line(screen, line_color, second_start, second_end, line_width)


    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()


# 160 - 540 (x)

# 220- 620 (y)