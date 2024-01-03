import pygame
import numpy as np

from melee_env.enums import UsedCharacter

pygame.init()


ICON_SIZE = 24
SPACING = 25
SCREEN_WIDTH, SCREEN_HEIGHT = (ICON_SIZE+SPACING) * 34, 850
WHITE = np.array([255, 255, 255])
BLACK = np.array([0, 0, 0])
BACKGROUND_COLOR = WHITE

DARKRED = np.array([130, 0, 0])


BABY_COLOR = np.array([51, 255, 0])
OLD_COLOR = DARKRED

axis_x = 80
axis_top = 45
axis_bottom = SCREEN_HEIGHT - 45 - 50

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("SSBM ELO Visualization")

def compute_text_offsets(player_idx, text, y):

    x_center = (player_idx+3) * (ICON_SIZE+SPACING) + ICON_SIZE // 2

    x_offset = x_center - text.get_width() // 2
    y_offset = y - 25

    return x_offset, y_offset


def ease_in_out_quad(t):
    return t**2 if t < 0.5 else -(t * (t - 2))


def get_font_for_elo(normalized_elo):

    rgb = DARKRED * normalized_elo

    font_size = 13 + int(normalized_elo * 10)

    return rgb, font_size


def get_font_for_age(normalized_age):
    # TODO : Visualize next resample !

    if normalized_age < 0.5:
        rgb = BABY_COLOR * (1 - normalized_age * 2) + BLACK * (normalized_age * 2)
    else:
        rgb = BLACK * (1 - (normalized_age-0.5) * 2) + DARKRED * ((normalized_age-0.5) * 2)

    font_size = 13 + int(normalized_age * 10)

    return rgb, font_size




# Function to draw the players' positions based on ELO scores
def draw_players(players):
    screen.fill(BACKGROUND_COLOR)

    elos = [player["last_elo"] for player in players]
    max_elo = max(elos)
    min_elo = min(elos)

    ages = [player["last_age"] for player in players]
    max_age = max(ages)
    min_age = min(ages)

    pygame.draw.line(screen, BLACK, (axis_x+5, axis_top), (axis_x+5, axis_bottom), 2)
    font = pygame.font.Font("seedsmash/visualization/fonts/Readable9x4.ttf", 20)
    legend_font = pygame.font.Font("seedsmash/visualization/fonts/Readable9x4.ttf", 30)
    text = legend_font.render("ELO", True, BLACK)
    text_rect = text.get_rect(midright=(axis_x - 10, 15))
    screen.blit(text, text_rect)

    text = legend_font.render("GAMES PLAYED", True, BLACK)
    text_rect = text.get_rect(midright=(SCREEN_WIDTH - 10, SCREEN_HEIGHT-15))
    screen.blit(text, text_rect)

    for player in players:
        x = (ICON_SIZE+SPACING)*(player["rank"]+3)
        normalized_elo = (player["last_elo"]-min_elo) / (max_elo-min_elo)
        normalized_age = (player["last_age"]-min_age) / (1 + max_age-min_age)

        y = axis_bottom - normalized_elo * (axis_bottom - axis_top)
        screen.blit(player["icon"], (x, y))

        rgb, font_size = get_font_for_elo(normalized_elo)
        elo_font = pygame.font.Font("seedsmash/visualization/fonts/Readable9x4.ttf", font_size)

        text = elo_font.render(str(int(player["last_elo"])), True, rgb)
        text_rect = text.get_rect(midright=(axis_x - 10, y))
        screen.blit(text, text_rect)

        rgb, font_size = get_font_for_age(normalized_age)
        age_font = pygame.font.Font("seedsmash/visualization/fonts/Readable9x4.ttf", font_size)

        text = age_font.render(str(int(player["last_age"])), True, rgb)
        text_rect = text.get_rect(center=(x+12, SCREEN_HEIGHT - 15-30))
        screen.blit(text, text_rect)

        text = font.render(player["name"].get(), True, BLACK)

        x_text, y_text = compute_text_offsets(player["rank"], text, y)
        screen.blit(text, (x_text, y_text))
        #screen.blit(text, (26*(i+1) + ICON_SIZE + 10, y + ICON_SIZE // 2 - text.get_height() // 2))

# Function to smoothly update the ELO scores based on the game results
def smooth_update_elo(players, frames=50):
    
    players = sorted(players, key=lambda p : -p["elo"]())
    
    for frame in range(frames):
        t = frame / frames
        easing = ease_in_out_quad(t)

        for rank, player in enumerate(players):
            diff = player["elo"]() - player["last_elo"]
            diff_rank = rank - player["rank"]
            diff_age = player["elo"].n - player["last_age"]

            player["last_elo"] = player["last_elo"] + easing * diff
            player["rank"] = player["rank"] + easing * diff_rank
            player["last_age"] = player["last_age"] + easing * diff_age


        draw_players(players)
        pygame.display.flip()
        pygame.time.delay(50)  # Adjust the delay (ms) to control animation speed

    for rank, player in enumerate(players):
        player["last_elo"] = player["elo"]()
        player["rank"] = rank
        player["last_age"] = player["elo"].n


if __name__ == '__main__':
    from seedsmash.ga.elo import Elo
    from seedsmash.ga.naming import Name
    players_ = [
        {"name": Name(),
         "elo": Elo(start=np.random.uniform(500, 1500)),
         "last_elo": 0.01,
         "last_age": 0,
         "icon": pygame.image.load(
             "seedsmash/visualization/features/icons/" + np.random.choice(list(UsedCharacter)).name + ".png"
         ),
         "rank": i}
    for i in range(30)]
    

    for p in players_:
        p["last_elo"] = p["elo"]()

    # Main game loop
    running = True
    clock = pygame.time.Clock()
    draw_players(players_)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Simulate game results here and update ELO scores
        # Example: smooth_update_elo(0, 1)  # Player 0 wins against Player 1
        matchups = np.random.choice(30, (256, 2), replace=True)
        for idx1, idx2 in matchups:
            p = idx1 / 30
            Elo.update_both(players_[idx1]["elo"], players_[idx2]["elo"], np.random.choice([0, 1], p=[p, 1-p]))

        smooth_update_elo(players_)
        # Draw the players' positions based on ELO scores

        pygame.display.flip()
        clock.tick(30)  # Set the frame rate (FPS)

    # Quit pygame
    pygame.quit()



