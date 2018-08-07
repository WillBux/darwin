from tqdm import trange
import random


class BasicSlaughterhouse:
    def __init__(self, template, population=1000, diversity_coef=0.4, max_diversity=20, decay=1.0, outlast=0.01, min_diversity=4):
        self.population = population
        self.diversity_coef = diversity_coef
        self.max_diversity = max_diversity
        self.min_diversity = min_diversity
        self.template = template
        self.decay = decay
        self.lr = template.lr
        self.beings = []
        self.parents = []
        self.outlast = outlast
        self.being_type = type(template)

    def create_generation(self):
        for i in range(self.population):
            self.beings.append([0, self.being_type(parents=[self.template], lr=self.lr, random=True)])

    def score_generation(self, x, y):
        for i in trange(len(self.beings)):
            self.beings[i][0] = self.beings[i][1].evaluate(x, y)
        self.beings = sorted(self.beings, key=lambda x: x[0], reverse=True)

    def next_generation(self):
        self.get_parents()
        self.beings = []
        for parent in self.parents:
            self.beings.append([0, parent])
        for i in range(self.population - len(self.parents)):
            if random.random() > self.diversity_coef:
                self.beings.append([0, self.being_type(parents=self.parents, lr=self.lr, random=False)])
            else:
                if random.random() > .5:
                    self.beings.append([0, self.being_type(
                        parents=[self.parents[int(random.random() * len(self.parents))]], lr=self.lr, random=False)])
                else:
                    self.beings.append([0, self.being_type(parents=[self.template], lr=self.lr, random=True)])

    def get_parents(self):
        self.parents = []
        high = self.beings[0][0]
        for i in range(self.max_diversity):
            if high - self.outlast < self.beings[i][0] or i < self.min_diversity:
                self.parents.append(self.beings[i][1])
                print('Parent Acc:', self.beings[i][0])

    def go(self, x, y, generations=1):
        if not self.beings:
            print('Start Generation')
            self.create_generation()
            self.score_generation(x, y)
        for i in range(generations):
            print('Generation ', i + 1)
            self.next_generation()
            self.score_generation(x, y)
            self.lr *= self.decay
        self.get_parents()
        return self.parents
