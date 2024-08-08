import pathlib
from collections import defaultdict

import pyglet
import numpy as np
from polaris.policies import PolicyParams
from pyglet.font.ttf import TruetypeInfo

from seedsmash2.submissions.bot_config import BotConfig
import os
file_path = pathlib.Path(__file__).parent.resolve()

assets_path = file_path / "assets/"

def load_fonts(path):
    for file in os.listdir(path):
        if file[-4:].lower() == ".ttf":
            filepath = os.path.join(path, file)
            p = TruetypeInfo(filepath)
            name = p.get_name("name")
            family = p.get_name('family')
            p.close()
            pyglet.font.add_file(filepath)
            print("Loaded font " + name + f'({family})' + " from " + filepath)


def compute_text_offsets(player_idx, text, y):
    x_center = (player_idx+3) * (RankingWindow.ICON_SIZE+RankingWindow.SPACING) + RankingWindow.ICON_SIZE // 2
    x_offset = x_center - text.width // 2
    y_offset = y - 25
    return x_offset, y_offset

def ease_in_out_quad(t):
    return t**2 if t < 0.5 else -(t * (t - 2))

def ease_in_out_sine(t):
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


class RankingWindow(pyglet.window.Window):
    WHITE = np.array([255, 255, 255, 255])
    BLACK = np.array([0, 0, 0, 255])
    BACKGROUND_COLOR = BLACK



    def __init__(self, players, *args, **kwargs):
        self.ICON_SIZE = 24
        self.SPACING = 35
        self.SCREEN_WIDTH = 400
        self.SCREEN_HEIGHT = 520
        self.MAX_PLAYERS = 15

        super().__init__(*args, **kwargs)
        self.batch = pyglet.graphics.Batch()

        self.video_set()

        pyglet.font.add_directory(str(assets_path / "fonts"))
        # Heavy font is loaded as bold....
        font = pyglet.font.load("A-OTF Folk Pro", 30, bold=True)


        self.sprites = {}
        self.rating_labels = {}
        self.player_name_labels = {}
        self.ranking_labels = {}
        self.bg_rects = {}
        self.icons = {}

        self.previous_stats = defaultdict(lambda : {
            "rank": 10,
            "rating": 0
        })

        self.data = [
            self.sprites, self.ranking_labels, self.player_name_labels, self.rating_labels, self.bg_rects,
            self.icons,
            self.previous_stats
        ]

        headfont = 13.5
        head_color = (190, 190, 190)

        x = 60
        y = self.SCREEN_HEIGHT - 24

        self.rating_label = pyglet.text.Label("Rating", font_name='A-OTF Folk Pro', font_size=headfont,
                          color=head_color, bold=True,
                          x=x + 250, y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)
        self.tag_label = pyglet.text.Label("Tag", font_name='A-OTF Folk Pro', font_size=headfont,
                          color=head_color, bold=True,
                          x=x +24, y=y , anchor_x='left', anchor_y='bottom', batch=self.batch)
        self.rank_label = pyglet.text.Label("Rank", font_name='A-OTF Folk Pro', font_size=headfont,
                          color=head_color, bold=True,
                          x=x - 32, y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)

        self.init = True


    def instanciate_for_player(self, player):
        x = 60

        y = self.SCREEN_HEIGHT - (self.previous_stats[player.name]["rank"] * 32 + 32)
        player_name = player.name

        main_character_img_path = assets_path / f"icons/{player.options.character.name}_{player.options.costume}.png"
        self.icons[player_name] = pyglet.image.load(main_character_img_path)

        self.sprites[player_name] = pyglet.sprite.Sprite(self.icons[player_name], x=x+220, y=y-2, batch=self.batch)

        rgb, font_size = get_font_for_rank(self.previous_stats[player.name]["rank"])

        self.rating_labels[player_name] = pyglet.text.Label(str(int(self.previous_stats[player.name]["rating"])), font_name='A-OTF Folk Pro', font_size=12,
                                      color=(255,255,255,255), bold=True,
                                      x=x+250 , y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)

        self.player_name_labels[player_name] = pyglet.text.Label(player_name, font_name='A-OTF Folk Pro', font_size=11,
                                              color=(255,255,255,255), bold=True,
                                              x=x+24, y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)

        self.ranking_labels[player_name] = pyglet.text.Label(str(self.previous_stats[player.name]["rank"]), font_name='A-OTF Folk Pro', font_size=font_size,
                                              color=rgb, bold=True,
                                              x=x-32, y=y-4, anchor_x='left', anchor_y='bottom', batch=self.batch)

        self.bg_rects[player_name] = pyglet.shapes.Rectangle(x-38, y-2, 360, 24, color=(20, 20, 20, 255),
                                                             batch=self.batch)




    def remove_player(self, player):
        # TODO: Does that remove the components from the batch ?
        tag = player.name
        for d in self.data:
            if tag in d:
                try:
                    d[tag].delete()
                except:
                    pass
                del d[tag]

    def update_ratings(self, policy_params, frames=40):

        if self.init:
            self.init = False
            # init (needed for the first icon, for some reason)
            dummy = PolicyParams(stats={"rank": 0, "rating": 0}, options=BotConfig())
            self.instanciate_for_player(dummy)
            self.remove_player(dummy)

        players = list(policy_params.values())
        for p in players:
            if p.stats['rank'] <= self.MAX_PLAYERS:
                if p.name not in self.player_name_labels:
                    self.instanciate_for_player(p)

            self.previous_stats[p.name]["old_rank"] = self.previous_stats[p.name]["rank"]

        for frame in range(frames+1):
            t = frame / frames
            easing = ease_in_out_sine(t)


            for i, player in enumerate(players):

                self.previous_stats[player.name]["rating"] = (easing * player.stats["rating"]
                                                              + (1-easing) * self.previous_stats[player.name]["rating"])
                self.previous_stats[player.name]["rank"] = (easing * player.stats["rank"] + (1-easing) *
                                                            self.previous_stats[player.name]["rank"])

            # we need to update the screen too
            #pyglet.gl.glClearColor(0, 0, 0, 1)  # Set the clear color to white
            self.clear()
            pyglet.clock.tick()
            # update params
            self.update_batch(players, t)
            self.batch.draw()
            #self.flip()


        for player in players:
            self.previous_stats[player.name]["rating"] = player.stats["rating"]
            self.previous_stats[player.name]["rank"] = player.stats["rank"]

            if player.stats["rank"] <= self.MAX_PLAYERS:
               pass
            else:
                self.remove_player(player)
        #
        # update params
        self.update_batch(players, 1)
        # we need to update the screen too
        pyglet.gl.glClearColor(0, 0, 0, 1)  # Set the clear color to white
        self.clear()
        pyglet.clock.tick()
        self.batch.draw()

    def update_batch(self, players, t):

        for player in players:
            player_name = player.name
            if player_name not in self.player_name_labels:
                continue

            x = 60
            bump_x = int(self.previous_stats[player.name]["old_rank"] - player.stats["rank"] < 0) * bump(t)
            x += int(25 * bump_x)

            # x = (self.ICON_SIZE + self.SPACING) * (player["rank"] + 3)
            y = self.SCREEN_HEIGHT - (self.previous_stats[player.name]["rank"] * 32 + 32)

            self.sprites[player_name].y = y - 2
            self.sprites[player_name].x = x + 220

            rgb, font_size = get_font_for_rank(self.previous_stats[player.name]["rank"])

            self.rating_labels[player_name].text = str(int(self.previous_stats[player.name]["rating"]))
            self.rating_labels[player_name].y = y
            self.rating_labels[player_name].x = x + 250

            self.player_name_labels[player_name].y = y
            self.player_name_labels[player_name].x = x + 24


            self.ranking_labels[player_name].text = str(round(self.previous_stats[player.name]["rank"]))
            self.ranking_labels[player_name].y = y - 4
            self.ranking_labels[player_name].x = x - 32

            self.bg_rects[player_name].x = x - 38
            self.bg_rects[player_name].y = y - 2


            self.ranking_labels[player_name].color = rgb
            self.ranking_labels[player_name].font_size = font_size

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
        self.set_caption("SeedSmash ranking")
        self.set_fullscreen(False)





if __name__ == '__main__':
    from seedsmash.ga.elo import Elo

    players_ = [
        {'config': BotConfig(tag="bob"), "rating": 0},
        {'config': BotConfig(tag="marth", character="MARTH", costume=2), "rating": 0},
        {'config': BotConfig(tag="jack", character="YLINK", costume="WHITE"), "rating": 0},
        {'config': BotConfig(tag="chevrifan420", character="FOX", costume=1), "rating": 0},
        {'config': BotConfig(tag="bobiij"), "rating": 0},
        {'config': BotConfig(tag="j", character="MARTH", costume=2), "rating": 0},
        {'config': BotConfig(tag="vspkv", character="YLINK", costume="WHITE"), "rating": 0},
        {'config': BotConfig(tag="dakork", character="LUIGI", costume=1), "rating": 0},
        {'config': BotConfig(tag="boba"), "rating": 0},
        {'config': BotConfig(tag="martha", character="MARTH", costume=2), "rating": 0},
        {'config': BotConfig(tag="jacka", character="YLINK", costume="WHITE"), "rating": 0},
        {'config': BotConfig(tag="chevraifan420", character="FOX", costume=1), "rating": 0},
        {'config': BotConfig(tag="bobiiaj"), "rating": 0},
        {'config': BotConfig(tag="ja", character="MARTH", costume=2), "rating": 0},
        {'config': BotConfig(tag="vsapkv", character="YLINK", costume="WHITE"), "rating": 0},
        {'config': BotConfig(tag="dakaork", character="LUIGI", costume=1), "rating": 0},
        {'config': BotConfig(tag="PAPAGANONWWW", character="GANONDORF", costume=2), "rating": 0},

    ]

    for idx, player in enumerate(players_):
        # TODO: fox splashes
        conf = player["config"]
        player["prev_rating"] = 0
        player["rank"] = idx + 1
        player["prev_rank"] = idx + 1

    window = RankingWindow(players_)


    def update(dt):
        # Simulate game results here and update ELO scores
        # Example: smooth_update_elo(0, 1)  # Player 0 wins against Player 1
        for player in players_:
            player["rating"] += np.random.randint(0, 100)

        window.update_ratings(players_)
        # Draw the players' positions based on ELO scores


    pyglet.clock.schedule_interval(update, 5)

    pyglet.app.run()




