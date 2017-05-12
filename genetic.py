#
# genetic.py
#

import random
from math import pi, sqrt
from simulation import Simulation

MAXIMIZE, MINIMIZE = 11, 22

class Individual(object):
    alleles = (0,1)
    length = 10
    seperator = ''
    optimization = MINIMIZE

    def __init__(self, chromosome=None):
        self.chromosome = chromosome or self._makechromosome()
        self.score = None  # set during evaluation
    
    def _makechromosome(self):
        "makes a chromosome from randomly selected alleles."
        return [random.choice(self.alleles) for gene in range(self.length)]

    def evaluate(self, optimum=None):
        "this method MUST be overridden to evaluate individual fitness score."
        return self.score
    
    def crossover(self, other):
        "override this method to use your preferred crossover method."
        return self._twopoint(other)
    
    def mutate(self, gene):
        "override this method to use your preferred mutation method."
        self._pick(gene) 
    
    # sample mutation method
    def _pick(self, gene):
        "chooses a random allele to replace this gene's allele."
        self.chromosome[gene] = random.choice(self.alleles)
    
    # sample crossover method
    def _twopoint(self, other):
        "creates offspring via two-point crossover between mates."
        left, right = self._pickpivots()
        def mate(p0, p1):
            chromosome = p0.chromosome[:]
            chromosome[left:right] = p1.chromosome[left:right]
            child = p0.__class__(chromosome)
            child._repair(p0, p1)
            return child
        return mate(self, other), mate(other, self)

    # some crossover helpers ...
    def _repair(self, parent1, parent2):
        "override this method, if necessary, to fix duplicated genes."
        pass

    def _pickpivots(self):
        left = random.randrange(1, self.length-2)
        right = random.randrange(left, self.length-1)
        return left, right

    #
    # other methods
    #
    def __repr__(self):
        "returns string representation of self"
        chromosome_str = ''
        for gene in self.chromosome:
                if gene:
                        chromosome_str += '1'
                else:
                        chromosome_str += '0'
        return '<%s chromosome="%s" score=%s>' % \
               (self.__class__.__name__,
                chromosome_str, self.score)

    def copy(self):
        twin = self.__class__(self.chromosome[:])
        twin.score = self.score
        return twin


class Environment(object):
    def __init__(self, kind, population=None, size=100, maxgenerations=100, \
                 generation=0, crossover_rate=0.90, mutation_rate=0.02, \
                 optimum=None):
        self.kind = kind
        self.size = size
        self.optimum = optimum
        self.population = population or self._makepopulation()
        for individual in self.population:
            individual.evaluate(self.optimum)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.maxgenerations = maxgenerations
        self.generation = generation
        self.report()

    def _makepopulation(self):
        return [self.kind() for individual in range(self.size)]
    
    def run(self):
        try:
            while not self._goal():
                self.step()
        except KeyboardInterrupt:
            pass

        best = self.best.copy()

        s = Simulation()
        while True:
            print(s.mySimul((best.chromosome[0], best.chromosome[1]), best.chromosome[2], best.chromosome[3],
                      best.chromosome[4], best.chromosome[5], best.chromosome[6], best.chromosome[7], best.chromosome[8]))

    
    def _goal(self):
        return self.generation >= self.maxgenerations or \
               self.best.score >= self.optimum
    
    def step(self):
        self.population.sort(key=lambda indiv: indiv.score, reverse=True)
        self._crossover()
        self.generation += 1
        self.report()
    
    def _crossover(self):
        next_population = [self.best.copy()]
        while len(next_population) < self.size:
            mate1 = self._select()
            if random.random() < self.crossover_rate:
                mate2 = self._select()
                offspring = mate1.crossover(mate2)
            else:
                offspring = [mate1.copy()]
            for individual in offspring:
                self._mutate(individual)
                individual.evaluate(self.optimum)
                next_population.append(individual)
        self.population = next_population[:self.size]

    def _select(self):
        "override this to use your preferred selection method"
        return self._tournament()
    
    def _mutate(self, individual):
        for gene in range(individual.length):
            if random.random() < self.mutation_rate:
                individual.mutate(gene)

    #
    # sample selection method
    #
    def _tournament(self, size=8, choosebest=0.90):
        competitors = [random.choice(self.population) for i in range(size)]
        competitors.sort(key=lambda indiv: indiv.score, reverse=True)
        if random.random() < choosebest:
            return competitors[0]
        else:
            return random.choice(competitors[1:])
    
    def best():
        doc = "individual with best fitness score in population."
        def fget(self):
            return self.population[0]
        return locals()
    best = property(**best())

    def report(self):
        print ("="*70)
        print ("generation: ", self.generation)
        print ("best:       ", self.best)


class MyIndividual(Individual):
    alleles = [(500, 600), (230, 350), (20, 80), (5, 40), (10, 45), (0, pi/2), (0, pi/2), (0, pi/2), (0, pi/2)]
    length = 9

    """
        pos_w -- the initial position
        pos_h -- the initial position
        ul -- the length of the upper leg
        ll -- the length of the lower leg
        w -- the width of the robot
        lua -- the angle of the left hip
        lla -- the angle of the left ankle
        rua -- the angle of the right hip
        rla -- the angle of the right ankle
    """


    def _makechromosome(self):
        "makes a chromosome from randomly selected alleles."
        return [random.uniform(self.alleles[gene][0], self.alleles[gene][1]) for gene in range(self.length)]


    # sample mutation method
    def _pick(self, gene):
        "chooses a random allele to replace this gene's allele."
        self.chromosome[gene] = random.uniform(self.alleles[gene][0], self.alleles[gene][1])


    def evaluate(self, optimum=None):
        "this method MUST be overridden to evaluate individual fitness score."

        s = Simulation(show=False)

        iterations, pos, ke = s.mySimul((self.chromosome[0], self.chromosome[1]), self.chromosome[2], self.chromosome[3],
                  self.chromosome[4], self.chromosome[5], self.chromosome[6], self.chromosome[7], self.chromosome[8])

        #dist = sqrt(pos[0]**2 + pos[1]**2)

        dist = pos[0]

        score = (600-dist)/6 # 0 - 100

        # score = ke**2 + iterations - dist**3

        self.score = score

    def __repr__(self):
        "returns string representation of self"
        chromosome_str = ''
        for gene in range (len(self.chromosome)):
            chromosome_str += ' ' + str(self.chromosome[gene])

        return '<%s chromosome="%s" \nscore=%s>' % \
               (self.__class__.__name__,
                chromosome_str, self.score)


e = Environment(MyIndividual, maxgenerations=200, mutation_rate=0.04, optimum=100)

e.run()

