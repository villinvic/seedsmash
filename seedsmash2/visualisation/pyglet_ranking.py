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
        self.SCREEN_WIDTH = 250
        self.SCREEN_HEIGHT = 1000

        super().__init__(*args, **kwargs)
        self.batch = pyglet.graphics.Batch()

        self.video_set()

        pyglet.font.load("fonts/A-OTF-FolkPro-Heavy.otf", 30)
        pyglet.font.load("fonts/A-OTF-FolkPro-Bold.otf", 30)


        self.sprites = {}
        self.rating_labels = {}
        self.player_name_labels = {}
        self.ranking_labels = {}

        for player in players:
            x = 40
            y = player["last_rank"] * 64 + 64
            player_name = player["config"].tag
            self.sprites[player["config"].tag] = pyglet.sprite.Sprite(player["icon"], x=x-10, y=y-2, batch=self.batch)

            rgb, font_size = get_font_for_rank(player["last_rank"])

            self.rating_labels[player_name] = pyglet.text.Label(str(int(player["last_rating"])), font_name='A-OTF-FolkPro-Bold', font_size=12,
                                          color=(255,255,255,255),
                                          x=x+150 , y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)

            self.player_name_labels[player_name] = pyglet.text.Label(player_name, font_name='A-<OTF-FolkPro-Heavy', font_size=12,
                                                  color=(255,255,255,255),
                                                  x=x+24, y=y, anchor_x='left', anchor_y='bottom', batch=self.batch)

            self.ranking_labels[player_name] = pyglet.text.Label(str(player["last_rank"]), font_name='A-OTF-FolkPro-Heavy', font_size=font_size,
                                                  color=rgb,
                                                  x=x-32, y=y-4, anchor_x='left', anchor_y='bottom', batch=self.batch)



    def update_ratings(self, players, frames=50):
        players_ranked = sorted(players, key=lambda p: -p["rating"])
        prev_rank = [
            p["last_rank"] for p in players_ranked
        ]
        for rank, p in enumerate(players_ranked, 1):
            p['rank'] = rank
        delta_rating = [
            [p["last_rating"], p["rating"]] for p in players_ranked
        ]

        for frame in range(frames+1):
            t = frame / frames
            easing = ease_in_out_quad(t)

            for rank, player in enumerate(players_ranked):

                player["last_rating"] = easing * delta_rating[rank][1] + (1-easing) * delta_rating[rank][0]
                player["last_rank"] =  easing * (rank+1) + (1-easing) * prev_rank[rank]

            # update params
            self.update_batch(players)
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
        self.update_batch(players)
        # we need to update the screen too
        pyglet.gl.glClearColor(0, 0, 0, 1)  # Set the clear color to white
        self.clear()
        pyglet.clock.tick()
        self.batch.draw()
        self.flip()

    def update_batch(self, players):

        for player in players:
            player_name = player["config"].tag

            # x = (self.ICON_SIZE + self.SPACING) * (player["rank"] + 3)
            y = round(- player["last_rank"] * 64 + 64)

            self.sprites[player_name].y = y -2

            rgb, font_size = get_font_for_rank(player["last_rank"])

            self.rating_labels[player_name].text = str(int(player["last_rating"]))
            self.rating_labels[player_name].y = y

            self.player_name_labels[player_name].y = y

            self.ranking_labels[player_name].text = str(round(player["last_rank"]))
            self.ranking_labels[player_name].y = y - 4
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


    pyglet.clock.schedule_interval(update, 1/30)

    pyglet.app.run()




