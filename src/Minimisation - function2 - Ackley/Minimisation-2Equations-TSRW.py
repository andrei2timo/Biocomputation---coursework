import random
import copy
import math
import matplotlib.pyplot as plt

# Each individual is represented by a data structure,
# consisting of an array of binary genes and a fitness value


class individual:
    gene = []
    fitness = 0

    def __repr__(self):
        return "Gene string " + "".join(str(x) for x in self.gene) + " - fitness: " + str(self.fitness)


# The initial population array of such individuals,
# and random gene length number of 10 and population of 50
P = 50
N = 10
GENERATIONS = 500  # initialise 500 generations
# random mutation rate and mutation step
MUTRATE = 0.03
MUTSTEP = 1.0

# Calculate individual's fitness
# first minimisation function


def mini_function(ind):
    fitness = 0
    first_exp = 0
    second_exp = 0
    for i in range(0, N):
        first_exp += math.cos(2 * math.pi * ind.gene[i])
        second_exp += ind.gene[i] * ind.gene[i]
    first_exp = math.exp((1/N) * first_exp)
    second_exp = (-20) * math.exp((-0.2) * math.sqrt((1/N) * second_exp))
    fitness = second_exp - first_exp
    return fitness

# Calculate population's fitness
# list parameter


def total_fitness(population):
    totalfit = 0
    for ind in population:
        totalfit += ind.fitness
    return totalfit

# Initialise original population


def initialise_population():
    population = []
    # Initialise population with random candidate solutions
    # Generate random genes and append to a temporary gene list
    # Assign fitness and gene to individual and append one to population
    for x in range(0, P):
        tempgene = []
        for x in range(0, N):
            # a random gene between -5.12 and 5.12(inclusive)
            tempgene.append(random.uniform(-32.0, 32.0))
        # print(tempgene)
        newindi = individual()  # initialise new instance
        # copy the gene from tempgene and assign to gene of individual
        newindi.gene = tempgene.copy()
        # initialise instance's fitness
        newindi.fitness = mini_function(newindi)
        population.append(newindi)

    return population

# Tourament Selection Process


def touranment_selection(population):
    offspring = []
    # Select two parents and recombine pairs of parents
    for i in range(0, P):
        parent1 = random.randint(0, P - 1)
        off1 = population[parent1]
        parent2 = random.randint(0, P - 1)
        off2 = population[parent2]
        # if one's fitness higher then add the smaller one to temp offsptring
        if (off1.fitness > off2.fitness):
            offspring.append(off2)
        else:
            offspring.append(off1)

    return offspring

# Roulette Wheel Selection Process


def RW_selection(population):
    # total fitness of initial pop
    total = 0
    for ind in population:
        total += abs(ind.fitness)

    offspring = []
    # Roulette Wheel Selection Process
    # Select two parents and recombine pairs of parents
    for i in range(0, P):
        selection_point = random.uniform(0.0, total)
        running_total = 0
        j = 0
        while running_total <= selection_point:
            running_total += abs(population[j].fitness)
            j += 1
            if(j == P):
                break

        # print(running_total)
        # print(j)
        offspring.append(copy.deepcopy(population[j-1]))

    return offspring

# Single-point Crossover process


def crossover(offspring):
    # Recombine pairs of parents from offspring
    crossover_offspring = []
    for i in range(0, P, 2):
        # pick up one random point in the gene length
        crosspoint = random.randint(1, N - 1)
        # 2 new temporary instances
        temp1 = individual()
        temp2 = individual()
        # 2 heads and 2 tails
        head1 = []
        tail1 = []
        head2 = []
        tail2 = []
        # print(offspring[i].gene, offspring[i+1].gene, crosspoint)
        # 0 to crosspoint adding gene to each head
        for j in range(0, crosspoint):
            head1.append(offspring[i].gene[j])
            head2.append(offspring[i + 1].gene[j])
        # crosspoint to N adding gene to each tail
        for j in range(crosspoint, N):
            tail2.append(offspring[i + 1].gene[j])
            tail1.append(offspring[i].gene[j])
        # print("head1 + tail2")
        # print(head1, tail2)
        # print("head2 + tail1")
        # print(head2, tail1)
        temp1.gene = head1 + tail2  # add first gene after crossover to temp1
        temp2.gene = head2 + tail1  # add second gene after crossover to temp2
        # call counting_ones to add fitness to temporary indv
        temp1.fitness = mini_function(temp1)
        temp2.fitness = mini_function(temp2)
        # append temp1, temp2 respectively to crosover_offspring_offspring
        crossover_offspring.append(temp1)
        crossover_offspring.append(temp2)
        # print(crosover_offspring_offspring[i].gene, crosover_offspring_offspring[i+1].gene)

    return crossover_offspring

# Bit-wise Mutation


def mutation(crossover_offspring, MUTRATE, MUTSTEP):
    # Mutate the result of new_offspring
    # Bit-wise Mutation
    mutate_offspring = []

    for i in range(0, P):
        new_indi = individual()
        new_indi.gene = []
        for j in range(0, N):
            gene = crossover_offspring[i].gene[j]
            ALTER = random.uniform(0.0, MUTSTEP)
            MUTPROB = random.uniform(0.0, 100.0)
            if (MUTPROB < (100*MUTRATE)):
                if(random.randint(0, 1) == 1):  # if random num is 1, add ALTER
                    gene += ALTER
                else:  # if random num is 0, minus ALTER
                    gene -= ALTER
                if(gene > 32.0):  # if gene value is larger than 32.0, reset it to 32.0
                    gene = 32.0
                if(gene < -32.0):  # if gene value is smaller than -32.0, reset it to -32.0
                    gene = -32.0
            new_indi.gene.append(gene)
        # add fitness to instance by calling mini_function
        new_indi.fitness = mini_function(new_indi)
        mutate_offspring.append(new_indi)

    return mutate_offspring

# Descending sorting


def sorting(population):
    #  descending sorting based on individual's fitness
    population.sort(key=lambda individual: individual.fitness, reverse=True)

    return population

# Minimisation Optimisation


def optimising(population, new_population):
    # more optimising
    # sorting instance with descending fitness
    population = sorting(population)

    # take the two instance with the worst fitness in the old population at index -1 and index -2
    worstFit_old_1 = population[-1]
    worstFit_old_2 = population[-2]

    # overwrite the old population with mutate_offspring
    population = copy.deepcopy(new_population)

    # sorting instance with descending fitness
    population = sorting(population)

    # after deepcopy new pop to old pop
    # take two instances with the best fitness in the new population at index 0 and index 1
    bestFit_new_1 = population[0]
    bestFit_new_2 = population[1]

    # compare the fitness btw the ones in the old pop and the ones in the new pop
    # replace the two best fitness/gene by the two worst fitness/gene at specific index in the new population
    if(worstFit_old_1.fitness < bestFit_new_1.fitness):
        population[0].gene = worstFit_old_1.gene
        population[0].fitness = worstFit_old_1.fitness
    if(worstFit_old_2.fitness < bestFit_new_2.fitness):
        population[1].gene = worstFit_old_2.gene
        population[1].fitness = worstFit_old_2.fitness

    return population


def GA(population, Selection, MUTRATE, MUTSTEP):
    # ===========GENETIC ALGORITHM===============
    # storing data to plot
    meanFit_values = []
    minFit_values = []

    for gen in range(0, GENERATIONS):
        # touranment selection process
        offspring = Selection(population)
        # crossover process
        crossover_offspring = crossover(offspring)
        # mutation process
        mutate_offspring = mutation(crossover_offspring, MUTRATE, MUTSTEP)
        # optimising
        population = optimising(population, mutate_offspring)

        # calculate Max and Mean Fitness
        # storing fitness in a list
        Fit = []
        for ind in population:
            Fit.append(mini_function(ind))
        # print(Fit)

        minFit = min(Fit)  # take out the min fitness among fitnesses in Fit
        meanFit = sum(Fit) / P  # sum all the fitness and divide by P size

        # append minFit and meanFit respectively to MinFit_values and MeanFit_values
        minFit_values.append(minFit)
        meanFit_values.append(meanFit)

        # # display
        # print("GENERATION " + str(gen + 1))
    print("Min Fitness: " + str(minFit) + "\n")
    print("Mean Fitness: " + str(meanFit) + "\n")

    return minFit_values, meanFit_values


# plotting
plt.ylabel("Fitness")
plt.xlabel("Number of Generation")

# Storing
minFit_data1 = []
minFit_data2 = []
minFit_data3 = []
minFit_data4 = []

meanFit_data1 = []
meanFit_data2 = []
meanFit_data3 = []
meanFit_data4 = []


# EXPERIMENT

# =============================================================
# TOURANMENT vs ROULETTE WHEEL SELECTION COMPARISON
# =============================================================

# [----------------- UNCOMMENT THIS AND ALTER N TO TEST -----------------]
N = 10
GENERATIONS = 500
plt.title("Minimisation GA \n Tournament and Roulette Wheel Selection \n"
          + "N = " + str(N) + " MUTRATE = " + str(MUTRATE) + " MUTSTEP = " + str(MUTSTEP))

# initialise original population
population = initialise_population()

minFit_data1, meanFit_data1 = GA(population, touranment_selection, 0.03, 1.0)
minFit_data2, meanFit_data2 = GA(population, RW_selection, 0.03, 1.0)

plt.plot(minFit_data1, label="Tournament")
plt.plot(minFit_data2, label="Roulette Wheel")

# [----------------- UNCOMMENT THIS AND ALTER N TO FOR TEST PURPOSES-----------------]

# N = 10
# MUTRATE = 0.03
# MUTSTEP = 1.0
# GENERATIONS = 500
# Min Fitness: -22.71253771450701 - TS
# Min Fitness: -22.518396025009025 - RW


# N = 20
# MUTRATE = 0.03
# MUTSTEP = 1.0
# GENERATIONS = 2000
# Min Fitness: -22.711649728948863 - TS
# Min Fitness: -22.615465913968446 - RW


# =============================================================
# TOURANMENT SELECTION
# =============================================================


# Best Fitness and Mean Fitness of TS
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Minimisation GA - Tournament Selection \n"
#           + "N = " + str(N) + " MUTRATE = " + str(MUTRATE) + " MUTSTEP = " + str(MUTSTEP))

# # initialise original population
# population = initialise_population()

# minFit_data1, meanFit_data1 = GA(population, touranment_selection, 0.03, 1.0)

# plt.plot(minFit_data1, label="Min Fitness")
# plt.plot(meanFit_data1, label="Mean Fitness")
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 10
# MUTRATE = 0.03
# MUTSTEP = 1.0
# Min Fitness: -22.698435758025173
# Mean Fitness: -22.425626443811986


# Vary MUTRATE
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Minimisation GA - Tournament Selection \n"
#           + "Vary MUTRATE")

# # initialise original population
# population = initialise_population()

# minFit_data1, meanFit_data1 = GA(population, touranment_selection, 0.3, 1.0)
# minFit_data2, meanFit_data2 = GA(population, touranment_selection, 0.03, 1.0)
# minFit_data3, meanFit_data3 = GA(population, touranment_selection, 0.003, 1.0)
# minFit_data4, meanFit_data4 = GA(population, touranment_selection, 0.0003, 1.0)

# plt.plot(minFit_data1, label="MUTRATE 0.3")
# plt.plot(minFit_data2, label="MUTRATE 0.03")
# plt.plot(minFit_data3, label="MUTRATE 0.003")
# plt.plot(minFit_data4, label="MUTRATE 0.0003")
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 10
# MUTSTEP = 1.0
# Min Fitness: -22.30083345122437 - MUTRATE 0.3
# Min Fitness: -22.69556884202363 - MUTRATE 0.03
# Min Fitness: -20.207411550234102 - MUTRATE 0.003
# Min Fitness: -11.77237542572478 - MUTRATE 0.0003


# =============================================================
# ROULETTE WHEEL SELECTION
# =============================================================


# Best Fitness and Mean Fitness of RW
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Minimisation GA - Roulette Wheel Selection \n"
#            + "N = " + str(N) + " MUTRATE = " + str(MUTRATE) + " MUTSTEP = " + str(MUTSTEP))

# # initialise original population
# population = initialise_population()

# minFit_data1, meanFit_data1 = GA(population, RW_selection, 0.03, 1.0)

# plt.plot(minFit_data1, label="Min Fitness")
# plt.plot(meanFit_data1, label="Mean Fitness")
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 10
# MUTRATE = 0.03
# MUTSTEP = 1.0
# Min Fitness: -22.65638596126548
# Mean Fitness: -21.481933208749926


# Vary MUTRATE
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Minimisation GA - Roulette Wheel Selection \n"
#           + "Vary MUTRATE")
# # initialise original population
# population = initialise_population()

# minFit_data1, meanFit_data1 = GA(population, RW_selection, 0.3, 1.0)
# minFit_data2, meanFit_data2 = GA(population, RW_selection, 0.03, 1.0)
# minFit_data3, meanFit_data3 = GA(population, RW_selection, 0.003, 1.0)
# minFit_data4, meanFit_data4 = GA(population, RW_selection, 0.0003, 1.0)

# plt.plot(minFit_data1, label="MUTRATE 0.3")
# plt.plot(minFit_data2, label="MUTRATE 0.03")
# plt.plot(minFit_data3, label="MUTRATE 0.003")
# plt.plot(minFit_data4, label="MUTRATE 0.0003")
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 10
# MUTSTEP = 0.3
# Min Fitness: -22.050692254005554 - MUTRATE 0.3
# Min Fitness: -22.49680201310099  - MUTRATE 0.03
# Min Fitness: -18.045801695277014 - MUTRATE 0.003
# Min Fitness: -10.085628544590312 - MUTRATE 0.0003


# DISPLAY PLOT
plt.legend(loc="upper right")
plt.show()
