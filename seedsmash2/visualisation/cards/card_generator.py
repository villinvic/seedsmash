import hashlib
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# look into this : http://slippicardgenerator.voidslayer.com/
# https://www.youtube.com/watch?v=0-3EmXgd_Wc&t=6s

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def color_distance(c1, c2):
    return np.sqrt(
        np.sum(np.square((c1-c2)/255))/3
    )

def attribute_color(bot_id):
    # Convert the player ID to a string and encode it
    player_id_str = str(bot_id + 777 / 12).encode()

    # Create a hash of the player ID
    hash_object = hashlib.sha256(player_id_str)
    hash_hex = hash_object.hexdigest()
    hash_int = int(hash_hex, 16)
    normalized_hash = (hash_int % (2**32))

    colors = dict(
    red = np.array([255, 5, 5], dtype=np.float32),
    green = np.array([0, 255, 5], dtype=np.float32),
    blue = np.array([5, 5, 255], dtype=np.float32),
    black = np.array([30, 30, 30], dtype=np.float32),
    white = np.array([255, 255, 255], dtype=np.float32),
    gold = np.array([212, 175, 55], dtype=np.float32),
    silver = np.array([192, 192, 192], dtype=np.float32),
    wild_card = np.array([0, 0, 0], dtype=np.float32),

    )

    color_names = list(colors.keys())
    weights = dict(
        red = 180,
        green = 180,
        blue = 180,
        black = 40,
        white = 15,
        gold = 3,
        silver = 1,
        wild_card=10,

    )
    max_weights = max(weights.values())
    min_weights = min(weights.values())

    def get_weights():
        total_weights = sum(weights.values())

        converge_probs = np.array([
            w / total_weights for c, w in weights.items()
        ])
        return converge_probs

    np.random.seed(normalized_hash)
    origin_color = np.random.choice(
            color_names, p=get_weights()
        )

    if origin_color in ["red", "blue", "green"]:
        deviation_color = np.random.randint(0, 255, 3)
        dev_w = 0.45
    elif origin_color == "white":
        deviation_color = np.random.randint(0, 255, 3)
        dev_w = 0.1
    elif origin_color == "wild_card":
        deviation_color = np.random.randint(20, 240, 3)

        while max([color_distance(deviation_color, color)
                   for color in colors.values()]) < 0.2:
            deviation_color = np.random.randint(20, 240, 3)
        dev_w = 1.0

    elif origin_color == "gold":
        deviation_color = np.random.randint(0, 255, 3)
        dev_w = 0.05
    elif origin_color == "black": # monochronic colors
        deviation_color = np.random.randint(0, 255)
        dev_w = 0.1
    else:
        deviation_color = np.random.randint(0, 255)
        dev_w = 0.2

    obtained_color = (1-dev_w) * colors[origin_color] + dev_w * deviation_color
    return np.uint8(obtained_color)


def create_diagonal_gradient_blend(card, stage):


    # Create diagonal gradient mask
    width, height = stage.size
    #gradient = Image.new("L", (width, height))
    #draw = ImageDraw.Draw(gradient)
    cx = int(width * 0.3)
    cy = int(3 * height / 5)

    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)
    g = (((x - cx) / width) ** 2 * 1.3 + ((cy - y) / height) ** 2 * 2.4)**0.2 * 1.15
    g = 1 - np.clip(g, 0.1, 1)
    gradient = (g * 255).astype(np.uint8)

    card.paste(stage, (int(-0.3*card.width), int(card.height*0.55-stage.height*0.5)), mask=Image.fromarray(gradient))

def blit_logo(image, bot_color):
    w = 150
    logo = inbue_with_bot_color(Image.open("assets/logo_alpha.png").resize((w, w)), bot_color,
                                0.85,
                                apply_black_power=5)


    alpha = np.float32(logo.split()[-1]) / 255
    c = w//2

    x = np.arange(0, w)
    x, y = np.meshgrid(x, x)

    d = (((x - c) / w) **2*0.4 + ((c - y) / w) ** 2)**1.5
    mini = np.min(d)
    maxi = np.max(d)
    d = (d - mini) / (maxi-mini)
    d = 1 - d

    gradient = (alpha * d * 255).astype(np.uint8)

    logo = ImageEnhance.Contrast(logo).enhance(0.9)


    image.paste(logo, (image.width-w, image.height-w), mask=Image.fromarray(gradient))

def inbue_with_bot_color(img, bot_color, weight, apply_black_power=2):
    pixels = np.float32(img)
    # apply more on darker pixels
    weight = np.maximum((1 - np.mean(pixels[:, :, :3], axis=2) / 255)**apply_black_power, 1e-2)[:, :, np.newaxis] * weight
    pixels[:, :, :3] = (1-weight) * pixels[:, :, :3] + weight * bot_color[np.newaxis, np.newaxis]
    return Image.fromarray(np.uint8(pixels))



def create_smash_player_card(bot_config, specialties, stats):
    char = bot_config["char"]
    stage = bot_config["stage"]
    color = bot_config["color"]
    nametag = bot_config["nametag"]

    BOT_COLOR = attribute_color(bot_config["bot_id"])

    main_character_img_path = f"assets/chars/{char.name}_{color}.png"
    stage_img_path = f"assets/stages/{stage.name}.png"
    # Load images
    character_img = Image.open(main_character_img_path).convert("RGBA")
    stage_img = inbue_with_bot_color(Image.open(stage_img_path).convert("RGBA"),
                                     BOT_COLOR,
                                     0.15,
                                     apply_black_power=4
                                     )
    card_background = inbue_with_bot_color(Image.open("assets/backgrounds/background_blue.jpeg").convert("RGBA"),
                                           BOT_COLOR,
                                           0.15,
                                           apply_black_power=5
                                           )

    slide = Image.open("assets/ui/slide.png")
    slide = slide.resize((int(slide.width*0.15), int(slide.height*0.15)),)
    slider = Image.open("assets/ui/slider.png")
    slider = slider.resize((int(slider.width*0.15), int(slider.height*0.15)),)


    # Create base card with the dimensions of the stage image
    width, height = card_background.size  # Get dimensions
    left = width * 0.7
    top = 0.15 * height
    right = width
    bottom = height * 0.85
    card = card_background.crop((left, top, right, bottom))
    create_diagonal_gradient_blend(card, stage_img)
    blit_logo(card, BOT_COLOR)
    width, height = card.size

    header = inbue_with_bot_color(Image.open("assets/ui/box1_gray.png"),BOT_COLOR,
                                           0.7,
                                           apply_black_power=0.5
                                           )
    header = header.resize((int(header.width*0.95), int(header.height*0.95)))
    alpha = np.minimum(np.float32(header.split()[-1]), 200).astype(np.uint8)
    card.paste(header, (width//2 - header.width//2, 0), Image.fromarray(alpha))

    # 136, 188
    character_img = character_img.resize((int(136*2), int(188*2)))
    alpha = np.float32(character_img.split()[-1])
    alpha[alpha < 32] = 0
    alpha[300:, :] *= np.linspace(1, 0, alpha.shape[0]-300)[:, np.newaxis]
    arr = np.float32(character_img)
    arr[:, :, 0] = 40*2
    arr[:, :, 1] = 40*2
    arr[:, :, 2] = 40*2
    #print(alpha[300:, :], alpha.shape)
    arr[300:, :, 3] *= np.linspace(1, 0, alpha.shape[0]-300)[:, np.newaxis]
    character_shadow = inbue_with_bot_color(Image.fromarray(np.uint8(arr)),
                                            BOT_COLOR,
                                            0.7,
                                            apply_black_power=0.5
                                            )

    alpha = np.minimum(alpha, 220).astype(np.uint8)

    char_x = int(width*0.77 - character_img.width /2)
    char_y = int(height*0.57 - character_img.height / 2)
    card.paste(character_shadow, (char_x, char_y), character_shadow)
    card.paste(character_img, (char_x, char_y), Image.fromarray(alpha))

    draw = ImageDraw.Draw(card)

    # Resize character image to fit on the card

    # Add nametag
    big = ImageFont.truetype("assets/fonts/A-OTF-FolkPro-Heavy.otf", 30)
    draw.text((int(width*0.6), int(height*0.1)), nametag, (255, 255, 255), font=big, anchor="ms")
    medium = ImageFont.truetype("assets/fonts/A-OTF-FolkPro-Heavy.otf", 22)
    small = ImageFont.truetype("assets/fonts/A-OTF-FolkPro-Heavy.otf", 18)
    tiny = ImageFont.truetype("assets/fonts/A-OTF-FolkPro-Heavy.otf", 14)


    # specialties
    x = int(width*0.05)
    y = int(height*0.85)
    target_value = np.array([250, 150, 26], dtype=np.float32)
    initial_value = np.array([100, 100, 255], dtype=np.float32)
    for i, (stat_name, value) in enumerate(specialties.items()):

        ratio = value / 100
        fill_color = initial_value * (1-ratio) + target_value * ratio

        draw.text((x, y + i * 20), stat_name, tuple(np.uint8(fill_color)), font=small)
        draw.text((x + 260, y + i * 20), str(value), tuple(np.uint8(fill_color)), font=small)

        slide_colored = np.float32(slide)
        slide_colored[:,:, :3] = fill_color[np.newaxis, np.newaxis]
        slider_colored = np.float32(slider)
        slider_colored[:, :, :3] = fill_color[np.newaxis, np.newaxis]

        #slide
        card.paste(Image.fromarray(np.uint8(slide_colored)), (x + 160, y + i * 20), slide)
        #slider
        dx = int(ratio * 65)
        card.paste(Image.fromarray(np.uint8(slider_colored)), (x + 160 + dx, y + i * 20), slider)

    # stats
    # main
    rank = stats["rank"]
    rating = stats["elo"]
    winrate = stats["winrate"]
    games_played = stats["games_played"]

    if rank == 1:
        fill_color = (212, 175, 55)
    elif rank == 2:
        fill_color = (192, 192, 192)
    elif rank == 3:
        fill_color = (205, 127, 50)
    else:
        fill_color = (255, 255, 255)

    draw.text((int(width*0.1), int(height*0.145)), "Rank", (255, 255, 255), font=medium)
    draw.text((int(width*0.1) + 72, int(height*0.145) - 5), str(rank), fill_color, font=big)

    draw.text((int(width*0.1), int(height*0.143) + 30), "Rating", (255, 255, 255), font=medium)
    draw.text((int(width*0.1) + 88, int(height*0.145) + 30), str(rating), (130, 130, 130), font=medium)

    draw.text((int(width * 0.1), int(height * 0.145) + 66), "Winrate", (255, 255, 255), font=medium)
    draw.text((int(width * 0.1) + 99, int(height * 0.145) + 66), f"{round(winrate*100)}%", (130, 130, 130), font=medium)

    draw.text((int(width * 0.45), int(height * 0.145)), "Games played", (255, 255, 255), font=medium)
    draw.text((int(width * 0.45) + 170, int(height * 0.145)), str(games_played), (130, 130, 130),font=medium)

    draw.text((int(width * 0.52), int(height * 0.145)+33), "Prefered stage", (255, 255, 255), font=medium)
    draw.text((int(width * 0.67), int(height * 0.145)+80), stage.name.replace("_", " "), (130, 130, 130), font=small, anchor="ms")


    # time

    # Define the desired format
    format_string = "%d.%m.%Y    %H:%M:%S"  # Example format

    # Format the date and time
    now = datetime.now()
    formatted_time = now.strftime(format_string)
    draw.text((int(width*0.14), int(height*0.315)), formatted_time, (130, 130, 130), font=medium)



    # Save the final card
    card.save(f"{nametag}_card.png")

from melee.enums import Stage, Character
# Example usage

stats = {
    "winrate": 0.78,
    "games_played": 4315,
    "elo": 2465,
    "rank": 2,
}

config = dict(
nametag = "Wario?",
color = "default",
char = Character.GAMEANDWATCH,
stage = Stage.YOSHIS_STORY,
bot_id = 19478888799
)

specialties = {
    "Patience": 85,
    "Agressivity": 64,
    "Combos": 91,
    "Off-stage plays": 74,
    "Swag": 80
}

create_smash_player_card(config, specialties, stats)
# obtained_colors = []
# for i in range(20):
#     obtained_colors.append(attribute_color(i))
#
#
# fig, ax = plt.subplots(figsize=(len(obtained_colors), 2))
# ax.set_title("obtained_colors", fontsize=16)
#
# # Create a bar for each color
# for idx, color in enumerate(obtained_colors):
#     ax.add_patch(plt.Rectangle((idx, 0), 1, 1, color=[c / 255.0 for c in color]))
#
# # Set limits and remove ticks
# ax.set_xlim(0, len(obtained_colors))
# ax.set_ylim(0, 1)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.savefig("obtained.color.png")