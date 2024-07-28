import pathlib

import pyglet
import numpy as np
import os
from melee_env.enums import UsedCharacter
from seedsmash2.submissions.bot_config import BotConfig

file_path = pathlib.Path(__file__).parent.resolve()

assets_path = file_path / "assets"
pyglet.resource.path = str(assets_path)
pyglet.resource.reindex()

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
        font_size = 20
    elif rank == 2:
        fill_color = (192, 192, 192, 255)
        font_size = 18
    elif rank == 3:
        fill_color = (205, 127, 50, 255)
        font_size = 16
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

        pyglet.font.load("fonts/A-OTF-FolkPro-Heavy.otf", 30)
        pyglet.font.load("fonts/A-OTF-FolkPro-Bold.otf", 30)


        self.sprites = {}
        self.rating_labels = {}
        self.player_name_labels = {}
        self.ranking_labels = {}
        self.data = [
            self.sprites, self.ranking_labels, self.player_name_labels, self.rating_labels
        ]

        for player in players_[:self.MAX_PLAYERS]:
            self.instanciate_for_player(player)

        headfont = 14
        head_color = tuple(np.int8(self.WHITE * 0.8))
        x = 60
        y = self.SCREEN_HEIGHT - 24
        pyglet.text.Label("Rating", font_name='A-OTF-FolkPro-Heavy', font_size=headfont,
                          color=head_color,
                          x=x + 250, y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)
        pyglet.text.Label("Tag", font_name='A-OTF-FolkPro-Heavy', font_size=headfont,
                          color=head_color,
                          x=x +24, y=y , anchor_x='left', anchor_y='bottom', batch=self.batch)
        pyglet.text.Label("Rank", font_name='A-OTF-FolkPro-Heavy', font_size=headfont,
                          color=head_color,
                          x=x - 32, y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)

    def instanciate_for_player(self, player):
        x = 60
        y = self.SCREEN_HEIGHT - (player["last_rank"] * 32 + 32)
        player_name = player["config"].tag
        self.sprites[player["config"].tag] = pyglet.sprite.Sprite(player["icon"], x=x+220, y=y-2, batch=self.batch)

        rgb, font_size = get_font_for_rank(player["last_rank"])

        self.rating_labels[player_name] = pyglet.text.Label(str(int(player["last_rating"])), font_name='A-OTF-FolkPro-Bold', font_size=12,
                                      color=(255,255,255,255),
                                      x=x+250 , y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)

        self.player_name_labels[player_name] = pyglet.text.Label(player_name, font_name='A-<OTF-FolkPro-Heavy', font_size=11,
                                              color=(255,255,255,255),
                                              x=x+24, y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)

        self.ranking_labels[player_name] = pyglet.text.Label(str(player["last_rank"]), font_name='A-OTF-FolkPro-Heavy', font_size=font_size,
                                              color=rgb,
                                              x=x-32, y=y-4, anchor_x='left', anchor_y='bottom', batch=self.batch)

    def remove_player(self, player):
        tag = player["config"].tag
        for d in self.data:
            if tag in d:
                d[tag].delete()
                del d[tag]

    def update_ratings(self, players, frames=60):

        # TODO: ranking will be obtained from trainer
        players_ranked = sorted(players, key=lambda p: -p["rating"])
        prev_rank = [
            p["last_rank"] for p in players_ranked
        ]
        for rank, p in enumerate(players_ranked, 1):
            p['rank'] = rank
            if rank <= self.MAX_PLAYERS:
                if p["config"].tag not in self.player_name_labels:
                    self.instanciate_for_player(p)
        delta_rating = [
            [p["last_rating"], p["rating"]] for p in players_ranked
        ]

        for frame in range(frames+1):
            t = frame / frames
            easing = ease_in_out_sine(t)

            for rank, player in enumerate(players_ranked):

                player["last_rating"] = easing * delta_rating[rank][1] + (1-easing) * delta_rating[rank][0]
                player["last_rank"] =  easing * (rank+1) + (1-easing) * prev_rank[rank]

            # update params
            self.update_batch(players, t)
            # we need to update the screen too
            pyglet.gl.glClearColor(0, 0, 0, 1)  # Set the clear color to white
            self.clear()
            pyglet.clock.tick()
            self.batch.draw()
            self.flip()


        for rank, player in enumerate(players_ranked):
            print(player["last_rank"], player["rank"], player["last_rating"], player["rating"])
            player["last_rating"] = player["rating"]
            player["rank"] = rank
        #
        # update params
        self.update_batch(players, 1)
        # we need to update the screen too
        pyglet.gl.glClearColor(0, 0, 0, 1)  # Set the clear color to white
        self.clear()
        pyglet.clock.tick()
        self.batch.draw()
        self.flip()

        for rank, p in enumerate(players_ranked, 1):
            p['rank'] = rank
            if rank <= self.MAX_PLAYERS:
               pass
            else:
                self.remove_player(p)

    def update_batch(self, players, t):

        for player in players:
            player_name = player["config"].tag
            if player_name not in self.player_name_labels:
                continue

            x = 60
            bump_x = int(player["last_rank"] - player["rank"] < -1e-8) * bump(t)
            x += int(25 * bump_x)

            # x = (self.ICON_SIZE + self.SPACING) * (player["rank"] + 3)
            y = self.SCREEN_HEIGHT - (player["last_rank"] * 32 + 32)

            self.sprites[player_name].y = y - 2
            self.sprites[player_name].x = x +220

            rgb, font_size = get_font_for_rank(player["last_rank"])

            self.rating_labels[player_name].text = str(int(player["last_rating"]))
            self.rating_labels[player_name].y = y
            self.rating_labels[player_name].x = x + 250

            self.player_name_labels[player_name].y = y
            self.player_name_labels[player_name].x = x + 24


            self.ranking_labels[player_name].text = str(round(player["last_rank"]))
            self.ranking_labels[player_name].y = y - 4
            self.ranking_labels[player_name].x = x - 32

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
        main_character_img_path = assets_path / f"icons/{conf.character.name}_{conf.costume}.png"
        player['icon'] = pyglet.image.load(main_character_img_path)
        player['icon'].scale = 1

        player["last_rating"] = 0
        player["rank"] = idx + 1
        player["last_rank"] = idx + 1

    window = RankingWindow(players_)


    def update(dt):
        # Simulate game results here and update ELO scores
        # Example: smooth_update_elo(0, 1)  # Player 0 wins against Player 1
        for player in players_:
            player["rating"] += np.random.randint(0, 100)

        window.update_ratings(players_)
        # Draw the players' positions based on ELO scores


    pyglet.clock.schedule_interval(update, 2)

    pyglet.app.run()




