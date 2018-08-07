import random
from math import exp, log1p
from darwin.metrics import *


class BinaryBeing:
    def __init__(self, features=None, variable_power=3, weight_sigma=5, lr=1, pruning=1, parents=None, metric='acc',
                 new=True):
        if parents is None:
            if features is None:
                raise ValueError('features cannot be initialized as None')
            self.features = features
            self.variable_power = variable_power
            self.weight_sigma = weight_sigma
            self.lr = lr
            self.pruning = pruning
            self.variables = []
            self.weights = []
            self.metric = metric
            self.e = []
            self.ln = []
            self.eW = []
            self.lnW = []
            self.setup()
        else:
            self.features = parents[0].features
            self.variable_power = parents[0].variable_power
            self.weight_sigma = parents[0].weight_sigma
            self.lr = lr
            self.pruning = parents[0].pruning
            self.metric = parents[0].metric
            self.variables = []
            self.weights = []
            self.e = []
            self.ln = []
            self.eW = []
            self.lnW = []
            if new:
                self.setup()
            else:
                self.mutate(parents)

    def setup(self):
        for i in range(int(self.features * self.variable_power * random.random() * 2 + 1)):
            temp = [random.randint(0, self.features - 1)]
            for j in range(int(self.variable_power * random.random())):
                temp.append(random.randint(0, self.features - 1))
            self.variables.append(temp)
        for i in range(len(self.variables)):
            if random.random() > .90:
                if random.random() > .5:
                    self.e.append(self.variables[i])
                else:
                    self.ln.append(self.variables[i])
        for i in range(len(self.variables)):
            self.weights.append(random.gauss(0, self.weight_sigma))
        for i in range(len(self.e)):
            self.eW.append(random.gauss(0, self.weight_sigma))
        for i in range(len(self.ln)):
            self.lnW.append(random.gauss(0, self.weight_sigma))

    def mutate(self, parents):
        for parent in parents:
            for var, weight in zip(parent.variables, parent.weights):
                if random.random() < self.pruning / len(parents):
                    self.variables.append(var)
                    self.weights.append(weight + random.gauss(0, self.lr))
            for e, eW in zip(parent.e, parent.eW):
                if random.random() < self.pruning / len(parents):
                    self.e.append(e)
                    self.eW.append(eW + random.gauss(0, self.lr))
            for ln, lnW in zip(parent.ln, parent.lnW):
                if random.random() < self.pruning / len(parents):
                    self.ln.append(ln)
                    self.lnW.append(lnW + random.gauss(0, self.lr))
        for i in range(int(self.pruning / random.random())):
            temp = [random.randint(0, self.features - 1)]
            for j in range(int(self.variable_power * random.random())):
                temp.append(random.randint(0, self.features - 1))
            if random.random() > .95:
                if random.random() > .5:
                    self.e.append(temp)
                    self.eW.append(random.gauss(0, self.weight_sigma))
                else:
                    self.ln.append(temp)
                    self.lnW.append(random.gauss(0, self.weight_sigma))
            self.variables.append(temp)
            self.weights.append(random.gauss(0, self.weight_sigma))

    def predict(self, x):
        pred = []
        for data in x:
            result = 0.0
            for var, weight in zip(self.variables, self.weights):
                mult = weight
                for elem in var:
                    mult *= data[elem]
                result += mult
            for e, eW in zip(self.e, self.eW):
                mult = 1
                for elem in e:
                    mult *= data[elem]
                if mult > 709:
                    result = 709
                result += eW * exp(mult)
            for ln, lnW in zip(self.ln, self.lnW):
                mult = 1
                for elem in ln:
                    mult *= data[elem]
                result += lnW * log1p(abs(mult))
            try:
                act = 1 / (1 + exp(-result))
            except OverflowError:
                act = 0.0
            pred.append(act)
        return pred

    def evaluate(self, x, y, metric=''):
        if metric == '':
            metric = self.metric
        pred = self.predict(x)
        if metric == 'acc':
            correct = 0
            incorrect = 0
            for y1, yp1 in zip(y, pred):
                if (yp1 >= .5 and y1 >= .5) or (yp1 < .5 and y1 < .5):
                    correct += 1
                else:
                    incorrect += 1
            return (correct + 0.0) / (correct + incorrect + 0.0)
        elif metric == 'roc':
            pred = np.nan_to_num(pred)
            return roc_auc_score(y, pred)


class CategoricalBeing:
    def __init__(self, features=None, classes=None, variable_power=3, weight_sigma=5, lr=1, pruning=1, parents=None,
                 metric='log_loss', new=True):
        if parents is None:
            if features is None:
                raise ValueError('features cannot be initialized as None')
            self.features = features
            self.classes = classes
            self.variable_power = variable_power
            self.weight_sigma = weight_sigma
            self.lr = lr
            self.pruning = pruning
            self.variables = []
            self.weights = []
            self.metric = metric
            self.e = []
            self.ln = []
            self.eW = []
            self.lnW = []
            self.setup()
        else:
            self.features = parents[0].features
            self.classes = parents[0].classes
            self.variable_power = parents[0].variable_power
            self.weight_sigma = parents[0].weight_sigma
            self.lr = lr
            self.pruning = parents[0].pruning
            self.metric = parents[0].metric
            self.variables = []
            self.weights = []
            self.e = []
            self.ln = []
            self.eW = []
            self.lnW = []
            if new:
                self.setup()
            else:
                self.mutate(parents)

    def setup(self):
        for j in range(self.classes):
            cv = []
            cvw = []
            ce = []
            cew = []
            cl = []
            clw = []
            for i in range(int(self.features * self.variable_power * random.random() * 2 + 1)):
                temp = [random.randint(0, self.features - 1)]
                for j in range(int(self.variable_power * random.random())):
                    temp.append(random.randint(0, self.features - 1))
                cv.append(temp)
            for i in range(len(cv)):
                if random.random() > .95:
                    if random.random() > .5:
                        ce.append(cv[i])
                    else:
                        cl.append(cv[i])
            for i in range(len(cv)):
                cvw.append(random.gauss(0, self.weight_sigma))
            for i in range(len(ce)):
                cew.append(random.gauss(0, self.weight_sigma))
            for i in range(len(cl)):
                clw.append(random.gauss(0, self.weight_sigma))
            self.variables.append(cv)
            self.weights.append(cvw)
            self.e.append(ce)
            self.eW.append(cew)
            self.ln.append(cl)
            self.lnW.append(clw)

    def mutate(self, parents):
        for i in range(self.classes):
            cv = []
            cvw = []
            ce = []
            cew = []
            cl = []
            clw = []
            for parent in parents:
                for var, weight in zip(parent.variables[i], parent.weights[i]):
                    if random.random() < self.pruning / len(parents):
                        cv.append(var)
                        cvw.append(weight + random.gauss(0, self.lr))
                for e, eW in zip(parent.e[i], parent.eW[i]):
                    if random.random() < self.pruning / len(parents):
                        ce.append(e)
                        cew.append(eW + random.gauss(0, self.lr))
                for ln, lnW in zip(parent.ln[i], parent.lnW[i]):
                    if random.random() < self.pruning / len(parents):
                        cl.append(ln)
                        clw.append(lnW + random.gauss(0, self.lr))
            for j in range(int(self.pruning / random.random())):
                temp = [random.randint(0, self.features - 1)]
                for q in range(int(self.variable_power * random.random())):
                    temp.append(random.randint(0, self.features - 1))
                if random.random() > .99:
                    if random.random() > .5:
                        ce.append(temp)
                        cew.append(random.gauss(0, self.weight_sigma))
                    else:
                        cl.append(temp)
                        clw.append(random.gauss(0, self.weight_sigma))
                cv.append(temp)
                cvw.append(random.gauss(0, self.weight_sigma))
            self.variables.append(cv)
            self.weights.append(cvw)
            self.e.append(ce)
            self.eW.append(cew)
            self.ln.append(cl)
            self.lnW.append(clw)

    def predict(self, x):
        pred = []
        for data in x:
            preAct = []
            for c in range(self.classes):
                result = 0.0
                for var, weight in zip(self.variables[c], self.weights[c]):
                    mult = weight
                    for elem in var:
                        mult *= data[elem]
                    result += mult
                for e, eW in zip(self.e[c], self.eW[c]):
                    mult = 1
                    for elem in e:
                        mult *= data[elem]
                    if mult > 709:
                        mult = 709
                    result += eW * exp(mult)
                for ln, lnW in zip(self.ln[c], self.lnW[c]):
                    mult = 1
                    for elem in ln:
                        mult *= data[elem]
                    result += lnW * log1p(abs(mult))
                if result > 709:
                    result = 709
                preAct.append(result)
            preAct_exp = [exp(p) for p in preAct]
            sum_preAct = sum(preAct_exp)
            if sum_preAct == 0:
                sum_preAct = 1
            softmax = [pe / sum_preAct for pe in preAct_exp]
            pred.append(softmax)
        return pred

    def evaluate(self, x, y, metric=''):
        if metric == '':
            metric = self.metric
        pred = self.predict(x)
        pred = np.nan_to_num(pred)
        if metric == 'log_loss':
            return 1 / log_loss(y, pred)
        elif metric == 'acc':
            y = np.array(y)
            pred = np.array(pred)
            correct = 0
            incorrect = 0
            for y1, yp1 in zip(y, pred):
                if np.argmax(y1) == np.argmax(yp1):
                    correct += 1
                else:
                    incorrect += 1
            return (correct + 0.0) / (incorrect + correct + 0.0)
