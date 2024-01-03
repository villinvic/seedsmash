import numpy as np


class Elo:
    def __init__(self, start=1000, k=40, k_next=20, old_age=50, base=400, locked=False):

        self.n = 0
        self.games_played = 0
        self.elite = False
        self.locked = locked

        self.gamma = 1/6.

        self.start = start
        self.current = start
        self.k = k
        self.k_next = k_next
        self.base = base
        self.old_age = old_age

    def update(self, other_elo, result):
        assert 0 <= result <= 1, result

        self.games_played += 1

        if not self.locked:
            d = self.k * (result - self.p(self() - other_elo()))

            self.current += d

            self.n += 1
            if self.n > self.old_age:
                self.k = self.k_next
            return d

    def p(self, distance):
        return 1. / (1. + 10**(-np.clip(distance, -15*self.base, 15*self.base)/self.base))

    def is_better_than(self, other):
        return self.current > other.current

    def __call__(self, *args, **kwargs):
        return self.current

    def win_prob(self, other_elo):
        if self() > other_elo():
            return self.p(self()-other_elo())
        else:
            return 1-self.p(self()-other_elo())

        #return np.exp(-(self.p(self.start-other_elo.current)-0.5)**2/(2*self.gamma**2))/ (np.sqrt(2*np.pi)*self.gamma)

    def __repr__(self):
        return f"Elo({self()})"


    @staticmethod
    def update_both(elo1, elo2, outcome):
        d = elo1.update(elo2, outcome)
        elo2.update(elo1, 1.-outcome)
        return d


if __name__ == '__main__':

    e = Elo()
    e2 = Elo(start=1100)
    e3 = Elo(start=1200)
    e4 = Elo(start=1300)
    e5 = Elo(start=1400)
    e6 = Elo(start=1500)
    e7 = Elo(start=1600)


    for other_elo in [e, e2, e3, e4, e5, e6, e7]:
        print(e.win_prob(other_elo), e())
        e.update(other_elo, 1.)
