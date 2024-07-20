from copy import deepcopy

import cv2
import numpy as np
import time
import random
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
FONT_SIZE = 32
FONT = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",FONT_SIZE)
#FONT = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
FONT_SCALE = 0.7
LINE_HEIGHT = 50
UPDATE_INTERVAL = 2
TRANSITION_DURATION = 1.0
ICON_SIZE = (40, 40)  # Size of the player icons
FPS = 30
players = [

    {"name": "MangoWWWW", "score": 0, "main": "FOX"},
    {"name": "Ludwig", "score": 0, "main": "FALCO"},
    {"name": "Charlie", "score": 0, "main": "MARTH"},
    {"name": "l", "score": 0, "main": "CPTFALCON"}
]



for player in players:
    player['icon'] = Image.open(f"assets/icons/{player['main']}.png").resize(ICON_SIZE, Image.ANTIALIAS)

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

def update_scores():
    for player in players:
        player["score"] += random.randint(0, 100)

def draw_rankings(img, data, positions, ranks, scores):
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for i, player in enumerate(data):
        x, y = positions[i]
        text = f"{round(ranks[i]):<3}     {player['name']:<20}|{int(scores[i]):<5}|"
        # cv2.putText(img, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA)

        icon_x = x + 35
        icon_y = y - ICON_SIZE[1] // 2  # Center the icon vertically
        pil_image.paste(player['icon'], (icon_x, icon_y), player['icon'])

        draw.text((x, y - FONT_SIZE // 2), text, font=FONT, fill=TEXT_COLOR)
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

def anneal(t, delta):
    start = delta[:, 0]
    end = delta[:, 1]
    return start + (end - start) * (1 - np.exp(-3 * t)) / (1 - np.exp(-3))

def bump(t, c):



def animate_transition(previous, current, duration, fps):
    frames = int(duration * fps)

    delta_rank = np.array([
        [prev["ranking"], curr["ranking"]] for prev, curr in zip(previous, current)
    ])

    delta_score = np.array([
        [prev["score"], curr["score"]] for prev, curr in zip(previous, current)
    ])

    def get_pos(rank):
        return (50, int(50 + rank * LINE_HEIGHT))

    for i in range(frames + 1):
        t = i / frames
        img = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, np.uint8)
        ranks = anneal(t, delta_rank)
        pos = [get_pos(rank) for rank in ranks]
        scores = anneal(t, delta_score)
        img =  draw_rankings(img, current, pos, ranks, scores)
        cv2.imshow("Ranking", img)
        #print(scores)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

cv2.namedWindow("Ranking")
previous_players = players
ranking = np.argsort([p["score"] for p in players])
for rank, idx in enumerate(ranking, 1):
    previous_players[idx]["ranking"] = rank

while True:
    start_time = time.time()

    update_scores()
    current_players = players
    ranking = np.argsort([p["score"] for p in players])
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