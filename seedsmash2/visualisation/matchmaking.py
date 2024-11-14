import datetime
import os
import pathlib
from collections import defaultdict

import pyglet
import numpy as np
from melee import Character
from polaris.policies import PolicyParams
from pyglet.font.ttf import TruetypeInfo

from melee_env.enums import UsedCharacter
from seedsmash2.submissions.bot_config import BotConfig
from seedsmash2.visualisation.cards.card_generator import create_smash_player_card



# 3 states:
# emoty
# rolling
# showing cards

file_path = pathlib.Path(__file__).parent.resolve()

assets_path = file_path / "assets/"

def ease_in_out_quad(t):

    if t < 0.2:
        return 0
    elif t > 0.9:
        return 1
    else:
        t = (t-.2) / 0.7

    return t**2 if t < 0.5 else -(t * (t - 2))

def ease_out_sine(t):
    if t < 0.2:
        return 0.
    elif t > 0.9:
        return 1.
    else:
        t = (t - 0.2)/(0.9-0.2)

    return -(np.cos(np.pi * t) - 1) / 2

def bump(t):
    return ((t - 0.5)**4 - 0.5**4) / (0.5**4)

def get_font_for_rank(rank):
    rank = round(rank)
    if rank == 1:
        fill_color = (212, 175, 55, 255)
        font_size = 18
    elif rank == 2:
        fill_color = (192, 192, 192, 255)
        font_size = 16
    elif rank == 3:
        fill_color = (205, 127, 50, 255)
        font_size = 15
    else:
        fill_color = (255, 255, 255, 255)
        font_size = 14
    return fill_color, font_size


class MatchMakingWindow(pyglet.window.Window):
    WHITE = np.array([255, 255, 255, 255])
    BLACK = np.array([0, 0, 0, 255])
    BACKGROUND_COLOR = WHITE

    def __init__(self, *args, **kwargs):
        self.ICON_SIZE = 24
        self.SPACING = 400
        self.LEFT_MARGIN = 780
        self.ANIMATION_WIDTH = 400
        self.SCREEN_WIDTH = self.ANIMATION_WIDTH + self.LEFT_MARGIN
        self.SCREEN_HEIGHT = 760
        super().__init__(*args, **kwargs)
        self.batch = pyglet.graphics.Batch()

        self.video_set()

        pyglet.font.add_directory(str(assets_path / "fonts"))
        # Heavy font is loaded as bold....
        font = pyglet.font.load("A-OTF Folk Pro", 30, bold=True)

        self.sprites = {}
        self.nametags = []
        self.nametags_bg = []
        self.player_name_labels = {}
        self.bg_rects = {}
        self.icons = {}

        self.selected = None

        self.data = [
            self.sprites, self.bg_rects,
            self.icons, self.player_name_labels
        ]

        headfont = 15
        head_color = (190, 190, 190)

        x = 60
        y = self.SCREEN_HEIGHT - 24

        self.init = True

        self.bg_rect_color = (80, 10, 10, 255)

    def instanciate_for_player(self, player):

        player_name = player.name
        main_character_img_path = assets_path / f"icons/{player.options.character.name}_{player.options.costume}.png"
        self.icons[player_name] = pyglet.image.load(main_character_img_path)

        self.sprites[player_name] = pyglet.sprite.Sprite(self.icons[player_name], x=-500, batch=self.batch)

        self.bg_rects[player_name] = pyglet.shapes.Rectangle(-500, 0, 60, 24, color=self.bg_rect_color,
                                                             batch=self.batch)
        self.player_name_labels[player_name] = pyglet.text.Label(player_name, font_name='A-OTF Folk Pro',
                                                                 font_size=11,
                                                                 color=(255, 255, 255, 255), bold=True,
                                                                 x=-500, anchor_x='left', anchor_y="bottom",
                                                                 batch=self.batch)

        vs_icon = assets_path / "ui/versus_red.png"
        img = pyglet.image.load(vs_icon)
        self.vs_icon = pyglet.sprite.Sprite(img, batch=self.batch)
        self.vs_icon.opacity = 0
        self.vs_icon.scale = 0.7

        self.nametags = [pyglet.text.Label("", font_name='A-OTF Folk Pro',
                                                                 font_size=12,
                                                                 color=(255, 255, 255, 255), bold=True,
                                                                 x=91 + i*196, y=50, anchor_x='center', anchor_y="bottom",
                                                                 batch=self.batch)
                                   for i in range(2)]
        self.nametags_bg = [pyglet.shapes.Rectangle(-500, 48, 84, 26, color=(100, 100, 100, 255),
                                                             batch=self.batch)
                         for i in range(2)]
        for c in self.nametags + self.nametags_bg:
            c.opacity = 0

        self.cards = []



    def remove_player(self, player):
        # TODO: Does that remove the components from the batch ?
        tag = player if isinstance(player, str) else player.name
        for d in self.data:
            if tag in d:
                try:
                    d[tag].delete()
                except:
                    pass
                del d[tag]

    def animate_matchmaking(self, selected, policy_params, frames=260):

        if self.init:
            self.init = False
            # init (needed for the first icon, for some reason)
            dummy = PolicyParams(stats={"rank": 0, "rating": 0}, options=BotConfig())
            self.instanciate_for_player(dummy)
            self.remove_player(dummy)

        params_copy = policy_params.copy()
        left_list = []
        right_list = []

        left_list.append(params_copy.pop(selected[0]))
        right_list.append(params_copy.pop(selected[1]))

        players = list(params_copy.values())
        np.random.shuffle(players)

        n = len(players)// 2
        left_list.extend(players[:n])
        right_list.extend(players[n:])


        for p in left_list+right_list:
            if p.name not in self.player_name_labels:
                self.instanciate_for_player(p)
        for pid in self.sprites:
            if pid not in policy_params:
                self.remove_player(pid)

        for card in self.cards:
            card.delete()
        del self.cards
        self.cards = []

        left_random_pos_start = np.random.uniform(0.95, 1.05)
        right_random_pos_start = np.random.uniform(0.95, 1.05)

        # we simulate the scrolling backwards.
        k = len(left_list)
        k2 = len(right_list)
        left_positions = np.zeros((k,))
        left_arange = np.arange(k)
        right_positions = np.zeros((k2,))
        right_arange = np.arange(k2)

        for p_label, bg, pid in zip(self.nametags, self.nametags_bg, selected):
            p_label.opacity = 0
            p_label.text = pid

            bg.opacity = 0
            bg.width = p_label.content_width + 12
            bg.x = p_label.x - bg.width//2


        selected = [policy_params[selected[0]], policy_params[selected[1]]]
        now = datetime.datetime.now()
        self.SPACING = 380 * ((k + k2) / 25)



        for frame in range(frames+1):
            if frame <= 200:
                t = frame / 200

                if t == 21 / 200:
                    for i, p in enumerate(selected):
                        card = create_smash_player_card(
                            p.options, p.stats, now_time=now
                        )
                        card_sprite = pyglet.sprite.Sprite(card, x=5+280 * i, y=40+95*i, batch=self.batch)
                        card_sprite.scale = 0.45
                        card_sprite.opacity = 0
                        self.cards.append(
                            card_sprite
                        )

                easing = ease_in_out_quad(1-t)

                left_positions[:] = (np.int32(easing * 4000 * left_random_pos_start + 100+left_arange*(200/k)) % 200)/200
                right_positions[:] = (np.int32(-easing * 4000 * right_random_pos_start + 100+right_arange*(200/k2)) % 200)/200
                self.clear()
                pyglet.clock.tick()
                self.scroll(t, left_list, left_positions, right_list, right_positions)
                self.batch.draw()

            else:
                t = (frame - 200) / 60

                self.clear()
                pyglet.clock.tick()
                self.show_cards(t)
                self.batch.draw()


    def scroll(self, t, left_players, left_positions, right_list, right_positions):

        fadein = 0
        fadeout = 0
        if t <= 0.05:
            fadein = (0.1 - t)/0.1
        elif t >= 0.99:
            fadeout = 1 - (1 - t) / 0.1


        self.vs_icon.opacity = round(255 * (1-fadein))
        self.vs_icon.x = self.LEFT_MARGIN//3 + self.ANIMATION_WIDTH //2 - self.vs_icon.width * 0.5
        self.vs_icon.y = self.SCREEN_HEIGHT //2 - self.vs_icon.height * 0.5
        self.vs_icon.scale = 0.7


        fading = np.maximum(fadein, fadeout)


        for lplayer, lpos in zip(left_players, left_positions):

            y = round((lpos-0.5) * self.SPACING)
            d_middle = 1 - np.minimum(np.abs(0.5-lpos) **0.33 * 1.3 , 1)
            c = d_middle * (1 - fading)

            self.sprites[lplayer.name].opacity = round(d_middle * 255 * (1-fading))
            self.sprites[lplayer.name].y = self.SCREEN_HEIGHT//2 + y - self.sprites[lplayer.name].height//2 + 1
            self.sprites[lplayer.name].x = self.LEFT_MARGIN//3 + self.ANIMATION_WIDTH //2 - 30 - self.sprites[lplayer.name].width//2

            self.player_name_labels[lplayer.name].y = (self.SCREEN_HEIGHT//2 + y -
                                                       self.player_name_labels[lplayer.name].content_height//2)
            self.player_name_labels[lplayer.name].x = (self.LEFT_MARGIN//3 + self.ANIMATION_WIDTH //2 - 50 -
                                                       self.player_name_labels[lplayer.name].content_width)
            self.player_name_labels[lplayer.name].color = (round(255*c), round(255*c), round(255*c), 255)

            self.bg_rects[lplayer.name].width = self.player_name_labels[lplayer.name].content_width + 38
            self.bg_rects[lplayer.name].y = (self.SCREEN_HEIGHT//2 + y -
                                                       self.player_name_labels[lplayer.name].content_height//2 - 4)
            self.bg_rects[lplayer.name].x = (self.LEFT_MARGIN//3 + self.ANIMATION_WIDTH //2 - 53 -
                                                       self.player_name_labels[lplayer.name].content_width)

            c = d_middle * (1-fading)
            self.bg_rects[lplayer.name].color = (round(80*c), round(10*c), round(10*c), 255)


        for rplayer, rpos in zip(right_list, right_positions):

            y = round((rpos-0.5) * self.SPACING)
            d_middle = 1 - np.minimum(np.abs(0.5-rpos) **0.33 * 1.3 , 1)
            c = d_middle * (1 - fading)


            self.sprites[rplayer.name].opacity = round(d_middle * 255 * (1-fading))
            self.sprites[rplayer.name].y = self.SCREEN_HEIGHT//2 + y - self.sprites[rplayer.name].height//2 + 1
            self.sprites[rplayer.name].x = self.LEFT_MARGIN//3 + self.ANIMATION_WIDTH //2 + 30 - self.sprites[rplayer.name].width//2

            self.player_name_labels[rplayer.name].y = (self.SCREEN_HEIGHT//2 + y -
                                                       self.player_name_labels[rplayer.name].content_height//2)
            self.player_name_labels[rplayer.name].x = (self.LEFT_MARGIN//3 + self.ANIMATION_WIDTH //2 + 50)
            self.player_name_labels[rplayer.name].color = (round(255*c), round(255*c), round(255*c), 255)

            self.bg_rects[rplayer.name].width = self.player_name_labels[rplayer.name].content_width + 38
            self.bg_rects[rplayer.name].y = (self.SCREEN_HEIGHT//2 + y -
                                                       self.player_name_labels[rplayer.name].content_height//2 - 4)
            self.bg_rects[rplayer.name].x = (self.LEFT_MARGIN//3 + self.ANIMATION_WIDTH //2 + 18)

            self.bg_rects[rplayer.name].color = (round(80*c), round(10*c), round(10*c), 255)
    def show_cards(self, t):
        x = 80
        if t <= 0.02:
            easing = t/0.02

        else:
            easing = 1

        if t <= 0.04:
            easing2 = t/0.04

        else:
            easing2 = 1


        self.vs_icon.scale = 0.7 + easing * 0.5
        self.vs_icon.x = (self.LEFT_MARGIN * easing2 + (1-easing2) * self.LEFT_MARGIN//3) + self.ANIMATION_WIDTH //2 - self.vs_icon.width * 0.5
        self.vs_icon.y = self.SCREEN_HEIGHT//2 - self.vs_icon.height * 0.5

        for p_label in self.nametags + self.nametags_bg:
            p_label.opacity = round(255*easing)

        for i, card in enumerate(self.cards):
            card.opacity = round(easing * 255)

            dy = -(1-easing2)*160

            if i == 1:
                dy = -dy

            card.x = self.LEFT_MARGIN + self.ANIMATION_WIDTH //2 - card.width * 0.5
            card.y = round(4 + 408 * (1-i) + dy)




    def fade_out(self):
        # smooth transition to black
        pass

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.flip()
    def video_set(self):
        self.default_display = pyglet.canvas.Display()
        self.default_screen = self.default_display.get_default_screen()
        self.set_size(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.location = self.default_screen.width // 2 - self.SCREEN_WIDTH // 2, self.default_screen.height // 2 - self.SCREEN_HEIGHT // 2
        self.set_location(self.location[0], self.location[1])
        self.set_caption("SeedSmash matchmaking")
        #pyglet.gl.glClearColor(0, 1, 0, 1)  # Note that these are values 0.0 - 1.0 and not (0-255).
        self.set_fullscreen(False)





if __name__ == '__main__':
    from seedsmash.ga.elo import Elo

    policies = {f"player_{i}": PolicyParams(
        name=f"player_{i}",
        options=BotConfig(tag=f"player_{i}", character=np.random.choice(UsedCharacter).name, costume=i % 3),
        stats={"rating": i, 'rank': 8, "winrate": 0.5, "games_played":1}
    ) for i in range(40)}

    window = MatchMakingWindow()

    t = 0
    def update(dt):

        # Simulate game results here and update ELO scores
        # Example: smooth_update_elo(0, 1)  # Player 0 wins against Player 1
        selected = np.random.choice(list(policies.keys()), size=2, replace=False)

        window.animate_matchmaking(selected, policies)
        # Draw the players' positions based on ELO scores

    pyglet.clock.schedule_interval(update, 12)

    pyglet.app.run()




