from copy import copy
from math import floor, ceil

import numpy as np

from nltk.corpus import words
import unidecode


class WordBag:
    words = set()

    @classmethod
    def sample(cls, n=1):
        words = np.random.choice(list(cls.words), n)
        if n == 1:
            return words[0]
        return words


class ProPlayers(WordBag):
    words = {
        'Nojima',
        'Hungrybox',
        'Leffen',
        'Mango',
        'Axe',
        'Wizzrobe',
        'Zain',
        'aMSa',
        'Plup',
        'iDBW',
        'Mew2King',
        'S2J',
        'Fiction',
        'SFAT',
        'moky',
        'n0ne',
        'Trif',
        'Captain Faceroll',
        'Swedish Delight',
        'Hax$',
        'Lucky',
        'Ginger',
        'Spark',
        'ChuDat',
        'PewPewU',
        'IIoD',
        'ARMY',
        'AbsentPage',
        'Bananas',
        'KJH',
        'Shroomed',
        'Westballz',
        'Medz',
        'MikeHaze',
        'Professor Pro',
        '2saint',
        'Gahtzu',
        'Albert',
        'Spud',
        'FatGoku',
        'Rishi',
        'Bimbo',
        'Magi',
        'Morsecode762',
        'Jakenshaken',
        'HugS',
        'Stango',
        'Zamu',
        'Drephen',
        'Michael',
        'Ice',
        'billybopeep',
        'La Luna',
        'Colbol',
        'Overtriforce',
        'Slox',
        'Kalamazhu',
        'Nickemwit',
        'Jerry',
        'Aura',
        'Nut',
        'Kalvar',
        'Polish',
        'Kevin Maples',
        'Bladewise',
        'Tai',
        'Squid',
        'Forrest',
        'Joyboy',
        'koDoRiN',
        'Ryan Ford',
        'Free Palestine',
        'Ryobeat',
        'Ka-Master',
        'Kũrv',
        'Frenzy',
        'MoG',
        'Boyd',
        'Cool Lime',
        'bobby big ballz',
        'Nintendude',
        'Franz',
        'Nicki',
        'lint',
        'King Momo',
        'TheRealThing',
        'Umarth',
        'Zeo',
        'Pricent',
        'Prince Abu',
        'Amsah',
        'Rocky',
        'Sharkz',
        'HTwa',
        'Kage',
        'Schythed',
        'Panda',
        'Soonsay',
        'TheSWOOPER',
        'Snowy',
        'Azen',
    }


class FunnyNames(WordBag):
    words = {
        'nihyru',
        'Syquo',
        'Goji',
        'Punchim',
        'Jolie-Edeweiss',
        'dark siko',
        'El Moustacho',
        'chevrier',
        'JV5'
        '0-to-SD',
        'SheikIsTrap',
        'MarthArm',
        'Jigglybuff',
        'Mario',
        'Luigi',
        'FD',
        'random fan420',
        'Georges',
        'Ludwig'
    }


class EnglishDictionary(WordBag):
    def __init__(self):
        english_words = words.words()
        # Filter out offensive words
        EnglishDictionary.words = {word for word in english_words if not word.islower() or word.isalpha()}

class Name:
    NAME_BAGS = {EnglishDictionary(): 0.7, FunnyNames(): 0.1, ProPlayers(): 0.2}
    origins = list(NAME_BAGS.keys())
    probs = list(NAME_BAGS.values())

    @staticmethod
    def valid_name(name):
        return 3 < len(name) < 15

    def __init__(self, name=None):
        if name is not None:
            self._name = name
        else:

            self._name = ""
            while not self.valid_name(self._name):
                self._name = ""
                components = [o.sample() for o in np.random.choice(Name.origins,
                                                                   2,
                                                                   p=Name.probs)]
                if len(components) > 1:
                    for i, c in enumerate(components):
                        point = np.int8(np.random.random() * (len(c)+1))

                        if np.random.random() < 0.5:
                            self._name += c[min(point, len(c)//2):]
                        else:
                            self._name += c[:max(point, len(c)//2)]

                        if np.random.random() < 0.1 and i < len(components) - 1:
                            char = np.random.choice([' ', '-'], p=[0.4, 0.6])
                            self._name += char
                else:
                    self._name = components[0][:20]

                self._name = self._name.strip().strip("_-")

    def inerit_from(self, *other_names):
        if len(other_names) == 1:
            addition = np.random.choice(Name.origins, p=Name.probs).sample()
        else:
            addition = other_names[1]._name


        self._name = ''
        while not self.valid_name(self._name):
            crossover_point1 = np.random.randint(1, len(other_names[0]._name) - 1)
            crossover_point2 = np.random.randint(1, len(addition) - 1)
            # Perform the crossover
            self._name = other_names[0]._name[:max(crossover_point1, len(other_names[0]._name)//2):] +\
                         addition[min(crossover_point2, len(addition)//2):]

    def get(self):
        return self._name

    def get_safe(self):
        return unidecode.unidecode(self._name.replace("$", ""))

    def set(self, name):
        self._name = name

    def __repr__(self):
        return self._name


if __name__ == '__main__':
    n = [Name() for _ in range(10)]

    n[0].inerit_from(n[1])

    n[2].inerit_from(n[3], n[4])


    print(n)
