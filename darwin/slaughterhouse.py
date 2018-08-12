from tqdm import trange
import random
from threading import Thread


class BasicSlaughterhouse:
    def __init__(self, template, population=1000, batch_size=32, diversity_coef=0.4, max_diversity=20, decay=1.0, outlast=0.01, min_diversity=4):
        self.population = population
        self.batch_size = batch_size
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

    def __create_generation(self, x, y):
        being = self.being_type(parents=[self.template], lr=self.lr, new=True)
        being.evaluate(x, y)
        self.beings.append(being)

    def create_generation(self, x, y):
        first = True
        for i in trange(self.population//self.batch_size + 1, desc='Random Generation', unit='batches'):
            num = self.population % self.batch_size if first else self.batch_size
            first = False
            threads = [Thread(target=self.__create_generation, args=(x, y)) for _ in range(num)]
            [thread.start() for thread in threads]
            [thread.join() for thread in threads]
        self.beings.sort(reverse=True)

    def __next_generation(self, x, y):
        if random.random() > self.diversity_coef:
            being = self.being_type(parents=self.parents, lr=self.lr, new=False)
        else:
            if random.random() > .5:
                being = self.being_type(parents=[self.parents[int(random.random() * len(self.parents))]],
                                        lr=self.lr, new=False)
            else:
                being = self.being_type(parents=[self.template], lr=self.lr, new=True)
        being.evaluate(x, y)
        self.beings.append(being)

    def next_generation(self, x, y, msg=''):
        self.get_parents()
        self.beings = []
        for parent in self.parents:
            self.beings.append(parent)
        create = self.population - len(self.parents)
        first = True
        for i in trange(create//self.batch_size, desc=msg, unit='batches'):
            num = create % self.batch_size if first else self.batch_size
            first = False
            threads = [Thread(target=self.__next_generation, args=(x, y)) for _ in range(num)]
            [thread.start() for thread in threads]
            [thread.join() for thread in threads]
        self.beings.sort(reverse=True)

    def get_parents(self):
        self.parents = []
        high = self.beings[0].score
        for i in range(self.max_diversity):
            if high - self.outlast < self.beings[i].score or i < self.min_diversity:
                self.parents.append(self.beings[i])
                print('Parent Acc:', self.beings[i].score)

    def go(self, x, y, generations=1):
        if not self.beings:
            self.create_generation(x, y)
        for i in range(generations):
            self.next_generation(x, y, msg=f'Generation {i+1}')
            self.lr *= self.decay
        self.get_parents()
        return self.parents
