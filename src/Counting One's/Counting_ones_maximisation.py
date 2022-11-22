import random
import copy
import matplotlib.pyplot as plt

# Each individual is represented by a data structure,
# consisting of an array of binary genes and a fitness value
class individual:
    gene = []
    fitness = 0

    def __repr__(self):
        return "Gene string " +  "".join(str(x) for x in self.gene) + " - fitness: " + str(self.fitness)

# The initial population of the individuals array,
# and a random gene length number of 50 and population of 50 
P = 50
N = 50
GENERATIONS = 100 # initialise 100 generations

# random mutation rate and mutation step
MUTRATE = 0.03
MUTSTEP = 0.9

# Calculate individual's fitness
# the individual's fitness is equal to the number of ‘1’s in its array of genes (genome)
# instance parameter
def counting_ones(ind):
    fitness = 0
    for i in range(0, N):
        # if(ind.gene[i] == 1): # if gene of an individual at index i equals to 1
            fitness = fitness + ind.gene[i]
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
            tempgene.append(random.uniform(0.0, 1.0)) # a random gene between 0.0 and 1.0(inclusive)
        # print(tempgene)
        newindi = individual() # initialise new instance
        newindi.gene = tempgene.copy() # copy the gene from tempgene and assign to gene of individual
        newindi.fitness = counting_ones(newindi) # initialise instance's fitness 
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
        if (off1.fitness > off2.fitness): # if one's fitness higher then add to temp offsptring
            offspring.append(off1)
        else:
            offspring.append(off2)

    return offspring

# Roulette Wheel Selection Process
def RW_selection(population):
    # total fitness of initial pop
    initial_fits = total_fitness(population)

    offspring = copy.deepcopy(population)
    # Roulette Wheel Selection Process
    # Select two parents and recombine pairs of parents
    for i in range(0, P):
        selection_point = random.uniform(0.0, initial_fits)
        running_total = 0
        j = 0
        while running_total <= selection_point:
            running_total += population[j].fitness
            j += 1
            if(j == P):
                break
        offspring[i] = population[j-1]

    return offspring

# Single-point Crossover process
def crossover(offspring):
    # Recombine pairs of parents from offspring
    crossover_offspring = []
    for i in range(0, P, 2):
        crosspoint = random.randint(0, N - 1) #pick up one random point in the gene length
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
        temp1.gene = head1 + tail2 # add first gene after crossover to temp1
        temp2.gene = head2 + tail1 # add second gene after crossover to temp2
        temp1.fitness = counting_ones(temp1) # call counting_ones to add fitness to temporary indv
        temp2.fitness = counting_ones(temp2)
        # append temp1, temp2 respectively to crosover_offspring_offspring
        crossover_offspring.append(temp1) 
        crossover_offspring.append(temp2)
        # print(crosover_offspring_offspring[i].gene, crosover_offspring_offspring[i+1].gene)

    return crossover_offspring

#Bit-wise Mutation
def mutation(crosover_offspring, MUTRATE, MUTSTEP):
    # Mutate the result of new_offspring
    mutate_offspring = []
    for i in range(0, P):
        new_indi = individual()
        new_indi.gene = []
        for j in range(0, N):
            gene = crosover_offspring[i].gene[j]
            ALTER = random.uniform(0.0, MUTSTEP)
            MUTPROB = random.uniform(0.0, 100.0)
            if (MUTPROB < (100*MUTRATE)):
                if(random.randint(0, 1) == 1): # if random num is 1, add ALTER
                    gene += ALTER
                else: # if random num is 0, minus ALTER
                    gene -= ALTER
                if(gene > 1.0): # if gene value is larger than 1.0, reset it to 1.0
                    gene = 1.0
                if(gene < 0.0): # if gene value is smaller than 0.0, reset it to 0.0
                    gene = 0.0
            new_indi.gene.append(gene) # add gene to instance
        new_indi.fitness = counting_ones(new_indi) # add fitness to instance by calling counting_ones
        mutate_offspring.append(new_indi)
    
    return mutate_offspring

# Descending sorting
def sorting(population):
    #  descending sorting based on  individual's fitness
    population.sort(key=lambda individual:individual.fitness, reverse=True)

    return population

# Optimisation
def optimising(population, new_population):
    # more optimising
    # sorting instance with descending fitness
    population = sorting(population)

    # take two instances with the best fitness in the old population at index 0 and index 1
    bestFit_old_1 = population[0]
    bestFit_old_2 = population[1]

    # overwrite the old population with mutate_offspring
    population = copy.deepcopy(new_population)

    # sorting instance with descending fitness
    population = sorting(population)

    # after deepcopy new pop to old pop
    # take the two instance with the worst fitness in the new population at index -1 and index -2
    worstFit_new_1 = population[-1]
    worstFit_new_2 = population[-2]

    # compare the fitness btw the ones in the old pop and the ones in the new pop
    # replace the two worst fitness/gene by the two best fitness/gene at specific index in the new population
    if(bestFit_old_1.fitness > worstFit_new_1.fitness):
        population[-1].fitness = bestFit_old_1.fitness
        population[-1].gene = bestFit_old_1.gene
    if(bestFit_old_2.fitness > worstFit_new_2.fitness):
        population[-2].fitness = bestFit_old_2.fitness
        population[-2].gene = bestFit_old_2.gene
   
    return population

def GA(population, Selection, MUTRATE, MUTSTEP):
    # storing data to plot
    meanFit_values = []
    maxFit_values = []
    # ===========GENETIC ALGORITHM===============
    
    for gen in range(0, GENERATIONS):
        # # touranment/ RW selection process
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
            Fit.append(counting_ones(ind))
        # print(Fit)

        maxFit = max(Fit) # take out the max fitness among fitnesses in Fit
        meanFit = sum(Fit)/ P # sum all the fitness and divide by Population size

        # append maxFit and meanFit respectively to MaxFit_values and MeanFit_values
        maxFit_values.append(maxFit)
        meanFit_values.append(meanFit)

        # display
        # print("GENERATION " + str(gen + 1))
        # print("Mean Fitness: " + str(meanFit)) 
        # print("Max Fitness: " + str(maxFit) + "\n")
    print("Max Fitness: " + str(maxFit) + "\n")
    print("Mean Fitness: " + str(meanFit) + "\n")
    
    return maxFit_values, meanFit_values

# plotting
plt.ylabel("Fitness")
plt.xlabel("Number of Generation")

#  Storing
maxFit_data1 = []
maxFit_data2 = []
maxFit_data3 = []
maxFit_data4 = []

meanFit_data1 = []
meanFit_data2 = []
meanFit_data3 = []
meanFit_data4 = []


# EXPERIMENT

# =============================================================
# TOURANMENT vs ROULETTE WHEEL SELECTION COMPARISON
# =============================================================

# [----------------- UNCOMMENT THIS AND ALTER N TO TEST -----------------]
#N = 50
plt.title("Maximisation GA \n Tournament and Roulette Wheel Selection \n" 
            + "N = " + str(N) + " MUTRATE = " + str(MUTRATE) + " MUTSTEP = " + str(MUTSTEP))

# initialise original population
population = initialise_population()

maxFit_data1, meanFit_data1 = GA(population, touranment_selection, 0.03, 0.9)
maxFit_data2, meanFit_data2 = GA(population, RW_selection, 0.03, 0.9)

plt.plot(maxFit_data1, label="Touranment")
plt.plot(maxFit_data2, label="Roulette Wheel")
# [----------------- UNCOMMENT THIS AND ALTER N TO TEST -----------------]

# N = 50
# MUTRATE = 0.03
# MUTSTEP = 0.9
# Max Fitness: 50.0 - TS
# Max Fitness: 46.62542552330858 - RW



# N = 100
# MUTRATE = 0.03
# MUTSTEP = 0.9
# Max Fitness: 98.06259568588314 - TS
# Max Fitness: 82.66273008007921 - RW



# =============================================================
# TOURANMENT SELECTION
# =============================================================


 #Best Fitness and Mean Fitness of TS
 #[----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Maximisation GA - Tournament Selection \n"
#             + "N = " + str(N) + " MUTRATE = " + str(MUTRATE) + " MUTSTEP = " + str(MUTSTEP))

# # initialise original population
# population = initialise_population()

# maxFit_data1, meanFit_data1 = GA(population, touranment_selection, 0.03, 0.9)

# plt.plot(maxFit_data1, label="Max Fitness")
# plt.plot(meanFit_data1, label="Mean Fitness")
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 50
# MUTRATE = 0.03
# MUTSTEP = 0.9
# Max Fitness: 50.0
# Mean Fitness: 49.467339709485785



# Vary MUTRATE
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# plt.title("Maximisation GA - Tournament Selection \n" 
#             + "Vary MUTRATE")

# # initialise original population
# population = initialise_population()

# maxFit_data1, meanFit_data1 = GA(population, touranment_selection, 0.3, 0.9)
# maxFit_data2, meanFit_data2 = GA(population, touranment_selection, 0.03, 0.9)
# maxFit_data3, meanFit_data3 = GA(population, touranment_selection, 0.003, 0.9)
# maxFit_data4, meanFit_data4 = GA(population, touranment_selection, 0.0003, 0.9)

# plt.plot(maxFit_data1, label="MUTRATE 0.3")
# plt.plot(maxFit_data2, label="MUTRATE 0.03")
# plt.plot(maxFit_data3, label="MUTRATE 0.003")
# plt.plot(maxFit_data4, label="MUTRATE 0.0003")
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 50
# MUTSTEP = 0.9
# Max Fitness: 43.471365866108314 - MUTRATE 0.3
# Max Fitness: 49.987334217326755 - MUTRATE 0.03
# Max Fitness: 48.887789618353786 - MUTRATE 0.003
# Max Fitness: 42.4534292474593   - MUTRATE 0.0003




# =============================================================
# ROULETTE WHEEL SELECTION
# =============================================================


# Best Fitness and Mean Fitness of RW
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# GENERATIONS = 200
# plt.title("Maximisation GA - Roulette Wheel Selection \n"
#            + "N = " + str(N) + " MUTRATE = " + str(MUTRATE) + " MUTSTEP = " + str(MUTSTEP))

# # initialise original population
# population = initialise_population()

# maxFit_data1, meanFit_data1 = GA(population, RW_selection, 0.03, 0.9)

# plt.plot(maxFit_data1, label="Max Fitness")
# plt.plot(meanFit_data1, label="Mean Fitness")
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 50
# MUTRATE = 0.03
# MUTSTEP = 0.9
# Max Fitness: 49.06776626442139
# Mean Fitness: 45.10946136458607


# Vary MUTRATE
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# GENERATIONS = 200
# plt.title("Maximisation GA - Roulette Wheel Selection \n" 
#             + "Vary MUTRATE")

# # initialise original population
# population = initialise_population()

# maxFit_data1, meanFit_data1 = GA(population, RW_selection, 0.3, 0.9)
# maxFit_data2, meanFit_data2 = GA(population, RW_selection, 0.03, 0.9)
# maxFit_data3, meanFit_data3 = GA(population, RW_selection, 0.003, 0.9)
# maxFit_data4, meanFit_data4 = GA(population, RW_selection, 0.0003, 0.9)

# plt.plot(maxFit_data1, label="MUTRATE 0.3")
# plt.plot(maxFit_data2, label="MUTRATE 0.03")
# plt.plot(maxFit_data3, label="MUTRATE 0.003")
# plt.plot(maxFit_data4, label="MUTRATE 0.0003")
# [----------------- UNCOMMENT THIS TO TEST -----------------]
# N = 50
# MUTSTEP = 0.9
# Max Fitness: 43.67294750948452 - MUTRATE 0.3
# Max Fitness: 49.18131827030025 - MUTRATE 0.03
# Max Fitness: 48.12886400473725 - MUTRATE 0.003
# Max Fitness: 40.591039436185326 - MUTRATE 0.0003

# DISPLAY PLOT
plt.legend(loc = "lower right")
plt.show()