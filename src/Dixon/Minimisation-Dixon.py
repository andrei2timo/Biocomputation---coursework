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
GENERATIONS = 600  # initialise 600 generations

# random mutation rate and mutation step
MUTRATE = 0.03
MUTSTEP = 1.0

# Calculate individual's fitness
def mini_function(ind):
    fitness = 0
    squared = 0
    first_expression = (ind.gene[0] - 1) ** 2 # (x1-1)^2
    second_expression = 0
    for i in range(1, N):
        # exp = [2 * xi^2 - x(i-1)]^2
        squared = i * ((2*ind.gene[i]*ind.gene[i] - ind.gene[i-1]) ** 2)
        # i * exp
        second_expression += squared 
    # (x1-1)^2 + SUM{ i* [2 * xi^2 - x(i-1)]^2 }
    fitness = first_expression + second_expression
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
            # a random gene between -10 and 10 (inclusive)
            tempgene.append(random.uniform(-10, 10))
        # print(tempgene)
        newindi = individual()  # initialise new instance
        # copy the gene from tempgene and assign to gene of individual
        newindi.gene = tempgene.copy()
        # initialise instance's fitness
        newindi.fitness = mini_function(newindi)
        population.append(newindi)
    return population

# Tourament Selection Method
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

# Roulette Wheel Selection Method
def RW_selection(population):
    # total fitness of initial pop
    total = 0
    for ind in population:
        total += 1/ind.fitness

    offspring = []
    # Roulette Wheel Selection Process
    # Select two parents and recombine pairs of parents
    for i in range(0, P):
        selection_point = random.uniform(0.0, total)
        running_total = 0
        j = 0
        while running_total <= selection_point:
            running_total += 1 / (population[j].fitness)
            j += 1
            if(j == P):
                break

        # print(running_total)
        # print(j)
        offspring.append(copy.deepcopy(population[j-1]))

    return offspring

# Single-point Crossover method
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

# Bit-wise Mutation method
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
                if(gene > 5.12):  # if gene value is larger than 5.12, reset it to 5.12
                    gene = 5.12
                if(gene < -5.12):  # if gene value is smaller than -5.12, reset it to -5.12
                    gene = -5.12
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
        # touranment selection process / RW selection process
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

        # display
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
# N = 10
# GENERATIONS = 600
plt.title("Minimisation GA \n Tournament and Roulette Wheel Selection \n"
        + "N = " + str(N) + " MUTRATE = " + str(MUTRATE) + " MUTSTEP = " + str(MUTSTEP))

# initialise original population
population = initialise_population()

minFit_data1, meanFit_data1 = GA(population, touranment_selection, 0.03, 1.0)
minFit_data2, meanFit_data2 = GA(population, RW_selection, 0.03, 1.0)

plt.plot(minFit_data1, label="Tournament")
plt.plot(minFit_data2, label="Roulette Wheel")
# [----------------- UNCOMMENT THIS AND ALTER N TO TEST -----------------]

# N = 10
# GENERATIONS = 600
# MUTRATE = 0.03
# MUTSTEP = 1.0
# Min Fitness: 0.007425401966637338 - TS
# Min Fitness: 0.0040744373402650386 - RW


# N = 10
# GENERATIONS = 2000
# MUTRATE = 0.03
# MUTSTEP = 1.0
# Min Fitness: 0.5000076286981088 - TS
# Min Fitness: 0.5000168729605512 - RW

# N = 50
# GENERATIONS = 4000
# MUTRATE = 0.03
# MUTSTEP = 1.0
# Min Fitness: 6.537547880219964 - TS
# Min Fitness: 0.8241292491869605 - RW


# =============================================================
# TOURANMENT SELECTION
# =============================================================


# Best Fitness and Mean Fitness of TS
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Minimisation GA - Tournament Selection \n"
#             + "N = " + str(N) + " MUTRATE = " + str(MUTRATE) + " MUTSTEP = " + str(MUTSTEP))

# # initialise original population
# population = initialise_population()

# minFit_data1, meanFit_data1 = GA(population, touranment_selection, 0.03, 1.0)

# plt.plot(minFit_data1, label="Min Fitness")
# plt.plot(meanFit_data1, label="Mean Fitness")

# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 10
# MUTRATE = 0.03
# MUTSTEP = 1.0
# Min Fitness: 0.011527590079116795
# Mean Fitness: 0.7425531247516733


# Vary MUTRATE
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Minimisation GA - Tournament Selection \n"
#             + "Vary MUTRATE")

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
# Min Fitness: 1.2020694193533954   - MUTRATE 0.3
# Min Fitness: 0.505488373883888 - MUTRATE 0.03
# Min Fitness: 1.9845575126780761   - MUTRATE 0.003
# Min Fitness: 9.81276143781134   - MUTRATE 0.0003


# =============================================================
# ROULETTE WHEEL SELECTION
# =============================================================


# Best Fitness and Mean Fitness of RW
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Minimisation GA - Roulette Wheel Selection \n"
#              + "N = " + str(N) + " MUTRATE = " + str(MUTRATE) + " MUTSTEP = " + str(MUTSTEP))

# # initialise original population
# population = initialise_population()

# minFit_data1, meanFit_data1 = GA(population, RW_selection, 0.03, 1.0)

# plt.plot(minFit_data1, label="Min Fitness")
# plt.plot(meanFit_data1, label="Mean Fitness")
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 10
# MUTRATE = 0.03
# MUTSTEP = 1.0
# Min Fitness: 0.03372919809378878
# Mean Fitness: 0.5167875596054593


# Vary MUTRATE
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Minimisation GA - Roulette Wheel Selection \n"
#            + "Vary MUTRATE")

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
# MUTSTEP = 1.0
# Min Fitness: 0.5449744249591794 -  MUTRATE 0.3
# Min Fitness: 0.5017947325596344  - MUTRATE 0.03
# Min Fitness: 0.7808550495952751 - MUTRATE 0.003
# Min Fitness: 96.18641960539154 - MUTRATE 0.0003


# DISPLAY PLOT
plt.legend(loc="upper right")
plt.show()
