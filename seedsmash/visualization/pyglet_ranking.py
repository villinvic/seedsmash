import pyglet
import numpy as np
import os
from melee_env.enums import UsedCharacter

current_script_dir = os.path.dirname(os.path.abspath(__file__))
pyglet.resource.path = current_script_dir
pyglet.resource.reindex()


def compute_text_offsets(player_idx, text, y):
    x_center = (player_idx+3) * (RankingWindow.ICON_SIZE+RankingWindow.SPACING) + RankingWindow.ICON_SIZE // 2
    x_offset = x_center - text.width // 2
    y_offset = y - 25
    return x_offset, y_offset

def ease_in_out_quad(t):
    return t**2 if t < 0.5 else -(t * (t - 2))

def get_font_for_elo(normalized_elo):
    rgb = RankingWindow.DARKRED * normalized_elo
    rgb[-1] = 255
    font_size = 12 + int(normalized_elo * 8)
    return tuple(np.int16(rgb)), font_size

def get_font_for_age(normalized_age):
    if normalized_age < 0.5:
        rgb = RankingWindow.BABY_COLOR * (1 - normalized_age * 2) + RankingWindow.BLACK * (normalized_age * 2)
    else:
        rgb = RankingWindow.BLACK * (1 - (normalized_age-0.5) * 2) + RankingWindow.DARKRED * ((normalized_age-0.5) * 2)
    font_size = 6 + int(normalized_age * 8)
    rgb[-1] = 255
    return tuple(np.int16(rgb)), font_size


class RankingWindow(pyglet.window.Window):
    WHITE = np.array([255, 255, 255, 255])
    BLACK = np.array([0, 0, 0, 255])
    BACKGROUND_COLOR = WHITE

    DARKRED = np.array([130, 0, 0, 255])

    BABY_COLOR = np.array([51, 255, 0, 255])
    OLD_COLOR = DARKRED



    def __init__(self, players, *args, **kwargs):
        self.ICON_SIZE = 24
        self.SPACING = 35
        self.axis_x = 85
        self.SCREEN_WIDTH = int(self.axis_x+(len(players)+1) * (self.ICON_SIZE + self.SPACING))
        self.SCREEN_HEIGHT = 900

        self.axis_top = self.SCREEN_HEIGHT - 65
        self.axis_bottom = 90

        self.char_icons = {
            char.name: pyglet.image.load(current_script_dir+f"/features/icons/{char.name}.png") for char in UsedCharacter
        }

        super().__init__(*args, **kwargs)
        self.batch = pyglet.graphics.Batch()

        self.video_set()

        self.y_axis = pyglet.shapes.Line(self.axis_x, self.axis_top, self.axis_x, self.axis_bottom, 2, color=(0,0,0,255), batch=self.batch)
        pyglet.font.load('fonts/Readable9x4', 30)

        self.elo_label = pyglet.text.Label("ELO Ranking", font_name="Readable9x4", font_size=25, color=(0,0,0,255), width=500,
                                      x=3, y=self.SCREEN_HEIGHT-15, anchor_x='left', anchor_y='center', batch=self.batch)

        self.games_played_label = pyglet.text.Label("Games played", font_name="Readable9x4", font_size=25, color=(0,0,0,255),
                                               x=self.SCREEN_WIDTH-10, y=20, anchor_x='right',
                                               anchor_y='center', batch=self.batch)

        elos = [player["last_elo"] for player in players.values()]
        max_elo = max(elos)
        min_elo = min(elos)

        ages = [player["last_age"] for player in players.values()]
        max_age = max(ages)
        min_age = min(ages)

        self.sprites = {}
        self.elo_labels = {}
        self.age_labels = {}
        self.player_name_labels = {}

        for player_name, player in players.items():
            x = int(self.axis_x + 10 + (self.SCREEN_WIDTH - 35 - self.axis_x - 10) * (player["rank"]) / len(players))
            normalized_elo = (player["last_elo"] - min_elo) / (max_elo - min_elo)
            normalized_age = (player["last_age"] - min_age) / (1 + max_age - min_age)

            y = int(self.axis_bottom + normalized_elo * (self.axis_top - self.axis_bottom))

            self.sprites[player_name] = pyglet.sprite.Sprite(self.char_icons[player["char"].name], x=x-12, y=y-12, batch=self.batch)

            rgb, font_size = get_font_for_elo(normalized_elo)

            self.elo_labels[player_name] = pyglet.text.Label(str(int(player["last_elo"])), font_name='Readable9x4', font_size=font_size,
                                          color=rgb,
                                          x=self.axis_x - 10, y=y, anchor_x='right', anchor_y='center', batch=self.batch)

            rgb, font_size = get_font_for_age(normalized_age)
            self.age_labels[player_name] = pyglet.text.Label(str(int(player["last_age"])), font_name='Readable9x4', font_size=font_size,
                                          color=rgb,
                                          x=x + 12, y= 15 + 30, anchor_x='center', anchor_y='center',
                                          batch=self.batch)

            self.player_name_labels[player_name] = pyglet.text.Label(player["name"], font_name='Readable9x4', font_size=12,
                                                  color=(0,0,0,255),
                                                  x=x, y=y+12, anchor_x='center', anchor_y='bottom', batch=self.batch)

    def update_elos(self, players, frames=50):
        players_ranked = sorted(list(players.values()), key=lambda p: -p["elo"]())

        for frame in range(frames):
            t = frame / frames
            easing = ease_in_out_quad(t)

            for rank, player in enumerate(players_ranked):
                diff = player["elo"]() - player["last_elo"]
                diff_rank = rank - player["rank"]
                diff_age = player["elo"].n - player["last_age"]

                player["last_elo"] = player["last_elo"] + easing * diff
                player["rank"] = player["rank"] + easing * diff_rank
                player["last_age"] = player["last_age"] + easing * diff_age


            # update params
            self.update_batch(players)
            # we need to update the screen too
            pyglet.gl.glClearColor(1, 1, 1, 1)  # Set the clear color to white
            self.clear()
            pyglet.clock.tick()
            self.batch.draw()
            self.flip()


        for rank, player in enumerate(players_ranked):
            player["last_elo"] = player["elo"]()
            player["rank"] = rank
            player["last_age"] = player["elo"].n

    def update_batch(self, players):
        # TODO : update when population is updated

        elos = [player["last_elo"] for player in players.values()]
        max_elo = max(elos)
        min_elo = min(elos)

        ages = [player["last_age"] for player in players.values()]
        max_age = max(ages)
        min_age = min(ages)

        for player_name, player in players.items():

            # x = (self.ICON_SIZE + self.SPACING) * (player["rank"] + 3)
            x = int(self.axis_x + self.SPACING*2 + (self.SCREEN_WIDTH - 3*self.SPACING - self.axis_x) * (player["rank"]) / len(players))

            normalized_elo = (player["last_elo"] - min_elo) / (max_elo - min_elo)
            normalized_age = (player["last_age"] - min_age) / (1 + max_age - min_age)

            y = int(self.axis_bottom + normalized_elo * (self.axis_top - self.axis_bottom))

            self.sprites[player_name].x = x-12
            self.sprites[player_name].y = y-12

            rgb, font_size = get_font_for_elo(normalized_elo)

            self.elo_labels[player_name].text = str(int(player["last_elo"]))

            self.elo_labels[player_name].color = rgb
            self.elo_labels[player_name].font_size = font_size
            self.elo_labels[player_name].y = y

            rgb, font_size = get_font_for_age(normalized_age)
            self.age_labels[player_name].text = str(int(player["last_age"]))
            self.age_labels[player_name].color = rgb
            self.age_labels[player_name].font_size = font_size
            self.age_labels[player_name].x = x

            self.player_name_labels[player_name].text = player["name"]
            self.player_name_labels[player_name].y = y+12
            self.player_name_labels[player_name].x = x

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
        self.set_caption("SSBM ELO Visualization")
        self.set_fullscreen(False)





if __name__ == '__main__':
    from seedsmash.ga.elo import Elo
    from seedsmash.ga.naming import Name

    players_ = [
        {"name": Name(),
         "elo": Elo(start=np.random.uniform(500, 1500)),
         "last_elo": 0.01,
         "last_age": 0,
         "char": np.random.choice(list(UsedCharacter)).name,
         "rank": i}
        for i in range(15)]

    for p in players_:
        p["last_elo"] = p["elo"]()

    print('bob')
    window = RankingWindow(players_)


    def update(dt):
        # Simulate game results here and update ELO scores
        # Example: smooth_update_elo(0, 1)  # Player 0 wins against Player 1
        matchups = np.random.choice(15, (256, 2), replace=True)
        for idx1, idx2 in matchups:
            p = idx1 / 30
            Elo.update_both(players_[idx1]["elo"], players_[idx2]["elo"], np.random.choice([0, 1], p=[p, 1 - p]))

        window.update_elos(players_)
        # Draw the players' positions based on ELO scores


    pyglet.clock.schedule_interval(update, 1/30)

    pyglet.app.run()




