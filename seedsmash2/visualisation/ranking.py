from copy import deepcopy

import cv2
import numpy as np
import time
import random
from PIL import Image, ImageDraw, ImageFont
import pathlib

from seedsmash2.submissions.bot_config import BotConfig


file_path = pathlib.Path(__file__).parent.resolve()

assets_path = file_path / "assets"


WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
FONT_SIZE = 32
FONT = ImageFont.truetype(str(assets_path / "fonts" / "A-OTF-FolkPro-Heavy.otf"),FONT_SIZE)
FONT_SCALE = 0.7
LINE_HEIGHT = 50
UPDATE_INTERVAL = 2

TRANSITION_DURATION = 0.2
ICON_SIZE = (40, 40)  # Size of the player icons
FPS = 60
players = [
    {'config': BotConfig(tag="bob"), "rating": 0},
    {'config': BotConfig(tag="marth", character="MARTH", costume=2), "rating": 0},
    {'config': BotConfig(tag="jack", character="YLINK", costume="WHITE"), "rating": 0},
    {'config': BotConfig(tag="chevrifan420", character="FOX", costume=1), "rating": 0},
]

for player in players:
    # TODO: fox splashes
    conf = player["config"]
    main_character_img_path = assets_path / f"icons/{conf.character.name}_{conf.costume}.png"
    player['icon'] = Image.open(main_character_img_path).resize(ICON_SIZE, Image.ANTIALIAS)

def draw_text_with_pillow(image, text, position, font, color):
    # Convert OpenCV image to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)


    draw.text(position, text, font=font, fill=color)
    # Convert PIL image back to OpenCV image
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def overlay_icon(img, icon, position):
    x, y = position
    y = int(-len(icon) // 1.5 + y)
    y1, y2 = y, y + icon.shape[0]
    x1, x2 = x, x + icon.shape[1]

    alpha_s = icon[:, :, 3] / 255.0
    alpha_s = alpha_s[:, :, np.newaxis]
    alpha_l = 1.0 - alpha_s

    img[y1:y2, x1:x2] = (alpha_s * icon[:, :, :3] + alpha_l * img[y1:y2, x1:x2])

def update_ratings():
    for player in players:
        player["rating"] += random.randint(0, 100)

def draw_rankings(img, data, positions, ranks, ratings):
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for i, player in enumerate(data):
        x, y = positions[i]
        tag = player["config"].tag
        # cv2.putText(img, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA)

        icon_x = x + 55
        icon_y = y - ICON_SIZE[1] // 2  # Center the icon vertically
        pil_image.paste(player['icon'], (icon_x, icon_y), player['icon'])

        draw.text((x, y - FONT_SIZE // 2), str(round(ranks[i])), font=FONT, fill=TEXT_COLOR)
        draw.text((x + 100, y - FONT_SIZE // 2), tag, font=FONT, fill=TEXT_COLOR)
        draw.text((x + 400, y - FONT_SIZE // 2), str(round(ratings[i])), font=FONT, fill=TEXT_COLOR)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)



def norm(x):
    # normalise x to range [-1,1]
    nom = (x - x.min()) * 1.0
    denom = x.max() - x.min()
    return  nom/denom

def sigmoid(x, k=0.1):
    # sigmoid function
    # use k to adjust the slope
    s = 1 / (1 + np.exp(-x / k))
    return s


def f(x):
    return np.exp(-1/(x+1e-8))

def anneal(t, delta):
    start = delta[:, 0]
    end = delta[:, 1]
    ft = f(t)
    return start + (end - start) * (ft / (ft + f(1-t)))

def bump(t):
    return (t - 0.5)**2 - 0.5**2


def animate_transition(previous, current, duration, fps):
    frames = int(duration * fps)

    delta_rank = np.array([
        [prev["ranking"], curr["ranking"]] for prev, curr in zip(previous, current)
    ])

    delta_rating = np.array([
        [prev["rating"], curr["rating"]] for prev, curr in zip(previous, current)
    ])

    def get_pos(t, r, d_rank):
        bump_dir = np.clip(np.diff(d_rank), -1, 0)
        return (50 + int(90 * bump(t) * bump_dir), int(50 + r * LINE_HEIGHT))

    for i in range(frames + 1):
        t = i / frames
        img = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, np.uint8)
        ranks = anneal(t, delta_rank)
        pos = [get_pos(t, rank, d_rank) for rank, d_rank in zip(ranks, delta_rank)]
        ratings = anneal(t, delta_rating)
        img =  draw_rankings(img, current, pos, ranks, ratings)
        cv2.imshow("Ranking", img)
        #print(ratings)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

cv2.namedWindow("Ranking")
previous_players = players
ranking = np.argsort([p["rating"] for p in players])
for rank, idx in enumerate(ranking, 1):
    previous_players[idx]["ranking"] = rank

while True:
    start_time = time.time()

    update_ratings()
    current_players = players
    ranking = np.argsort([-p["rating"] for p in players])
    for rank, idx in enumerate(ranking, 1):
        current_players[idx]["ranking"] = rank
    animate_transition(previous_players, current_players, TRANSITION_DURATION, FPS)
    previous_players = deepcopy(current_players)
    elapsed_time = time.time() - start_time
    if elapsed_time < UPDATE_INTERVAL:
        time.sleep(UPDATE_INTERVAL - elapsed_time)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()