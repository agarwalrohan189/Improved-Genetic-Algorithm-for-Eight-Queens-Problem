import random as rn
import numpy as np
from numpy.random import choice
import math
import matplotlib.pyplot as plt
rn.seed(13)


def diag_attacks(state):
    n = len(state)

    ans = 0
    for sum in range(1, 2*n+1):
        count = 0
        for i in range(n):
            if(i+1+int(state[i]) == sum):
                count += 1
        ans += count*(count-1)/2

    for diff in range(1-n, n):
        count = 0
        for i in range(n):
            if((int(state[i])-(i+1)) == diff):
                count += 1
        ans += count*(count-1)/2

    return ans


def row_attacks(state):
    n = len(state)

    ans = 0
    for row in range(1, n+1):
        count = 0
        for i in range(n):
            if(int(state[i]) == row):
                count += 1
        ans += count*(count-1)/2

    return ans


def calc_fitness(population):
    m = len(population[0])
    fitness = []
    for state in population:
        fitness.append(1+(m*(m-1)/2 - diag_attacks(state) -
                          row_attacks(state)))

    return fitness


def random_selecion(population, fitness, number=2):
    n = len(fitness)
    sum_of_fitness = np.sum(fitness)
    probabilities = [fitness[i]/sum_of_fitness for i in range(n)]
    draw = choice(population, number, p=probabilities, replace=False)
    return draw


def reproduce(x, y):
    n = len(x)
    br = rn.randint(0, n-1)
    child = ""
    for i in range(n):
        if(i <= br):
            child += x[i]
        else:
            child += y[i]
    return child


def mutate(z):
    n = len(z)
    pos = rn.randint(0, n-1)
    mutated = ""
    for i in range(n):
        if(i == pos):
            mutated += str(rn.randint(1, n))
        else:
            mutated += z[i]
    return mutated


def genetic_algo(population):
    new_population = []
    n = len(population)
    fitness = calc_fitness(population)
    for _ in range(n):
        x, y = random_selecion(population, fitness)
        z = reproduce(x, y)
        if(rn.random() > 0.8):
            z = mutate(z)
        new_population.append(z)
    return new_population


def random_selecion_improved(population, fitness, number=2):
    n = len(fitness)
    temp = [math.exp(f) for f in fitness]
    sum_of_fitness = np.sum(temp)
    probabilities = [temp[i]/sum_of_fitness for i in range(n)]
    draw = choice(population, number, p=probabilities, replace=False)
    return draw


def reproduce_improved(x, y):
    n = len(x)
    break1 = rn.randint(0, n-1)
    break2 = rn.randint(break1, n-1)
    child1 = ""
    child2 = ""
    child3 = ""
    child4 = ""
    child5 = ""
    child6 = ""
    for i in range(n):
        if(i <= break1):
            child1 += x[i]
            child2 += x[i]
            child3 += y[i]
            child4 += x[i]
            child5 += y[i]
            child6 += y[i]
        elif(i > break1 and i <= break2):
            child1 += x[i]
            child2 += y[i]
            child3 += x[i]
            child4 += y[i]
            child5 += x[i]
            child6 += y[i]
        else:
            child1 += y[i]
            child2 += x[i]
            child3 += x[i]
            child4 += y[i]
            child5 += y[i]
            child6 += x[i]

    children = [child1, child2, child3, child4, child5, child6]
    fitness = calc_fitness(children)
    selected_child = children[fitness.index(max(fitness))]
    return selected_child


def genetic_algo_improved(population):
    new_population = []
    n = len(population)
    fitness = calc_fitness(population)
    for _ in range(n):
        x, y = random_selecion_improved(population, fitness, 2)
        z = reproduce_improved(x, y)
        if rn.random() > 0.2:
            z = mutate(z)
            if rn.random() > 0.7:
                z = mutate(z)
        new_population.append(z)
    return new_population


def main():
    # set the maximum number of iterations
    max_iter = 5000

    GA_fitness = []
    state = "44444444"
    population = []
    for i in range(20):
        population.append(state)

    for gen in range(max_iter):
        print("Generation: ", (gen+1), "( Textbook Version )")

        population = genetic_algo(population)
        fitness = calc_fitness(population)
        max_fitness = 0
        state_with_max_fitness = []
        for i in range(20):
            if(fitness[i] > max_fitness):
                max_fitness = fitness[i]
                state_with_max_fitness = population[i]

        GA_fitness.append(max_fitness)
        print("Max Fitness: ", max_fitness,
              "state with max fitness: ", state_with_max_fitness, end='\n\n')

        if(max_fitness == 29):
            break

    if gen == max_iter:
        print("DID NOT CONVERGE TO A SOLUTION")

    print("----------------------------------------------------------------")
    print("Genetic Algo-improved")
    print("----------------------------------------------------------------")

    GA_fitness_improved = []
    state = "44444444"
    population = []
    for i in range(20):
        population.append(state)

    for gen_improved in range(max_iter):
        print("Generation: ", (gen_improved+1), "( Improved Version )")

        population = genetic_algo_improved(population)
        fitness = calc_fitness(population)
        max_fitness_improved = 0
        state_with_max_fitness_improved = []
        for i in range(20):
            if(fitness[i] > max_fitness_improved):
                max_fitness_improved = fitness[i]
                state_with_max_fitness_improved = population[i]

        GA_fitness_improved.append(max_fitness_improved)

        print("Max Fitness: ", max_fitness_improved,
              "state with max fitness: ", state_with_max_fitness_improved, end='\n\n')

        if(max_fitness_improved == 29):
            break

    if gen_improved == max_iter:
        print("DID NOT CONVERGE TO A SOLUTION")

    print("Final Results:- ")
    print("For textbook version", "Generation: ", (gen+1), "Max Fitness: ", max_fitness,
          "state with max fitness: ", state_with_max_fitness)

    print("For improved version", "Generation: ", (gen_improved+1), "Max Fitness: ", max_fitness_improved,
          "state with max fitness: ", state_with_max_fitness_improved)

    plt.plot(GA_fitness, label="Textbook Version", linewidth=2,)
    plt.plot(GA_fitness_improved, label="Improved Version",
             linewidth=2, )
    plt.ylabel('Best Fitness value in the population')
    plt.xlabel('Number of generations')
    plt.legend()
    plt.title('Comparison of Textbook and Improved Version for Eight Queens Problem')
    plt.show()

    print('\n\n')

    # This commented section is for the average performance of the algorithms for the Eight queens problem

    # num_epochs=10
    # max_iter=7000

    # print('\n\n------------------------------------------ TEXTBOOK VERSION -------------------------------------------------\n\n')

    # gen_sum = 0
    # gen_not_conv = 0
    # max_gen = 0
    # min_gen = 10000
    # gen_conv = []
    # for epoch in range(num_epochs):
    #     state = "44444444"
    #     population = []
    #     for i in range(20):
    #         population.append(state)
    #     gen = max_iter
    #     for itr in range(max_iter):
    #         population = genetic_algo(population)
    #         fitness = calc_fitness(population)
    #         max_fitness = 0
    #         state_with_max_fitness = []
    #         for i in range(20):
    #             if(fitness[i] > max_fitness):
    #                 max_fitness = fitness[i]
    #                 state_with_max_fitness = population[i]

    #         # print("Max Fitness: ", max_fitness,
    #         #       "state with max fitness: ", state_with_max_fitness)

    #         if(max_fitness == 29):
    #             gen = itr+1
    #             gen_sum += gen
    #             max_gen = max(max_gen, gen)
    #             min_gen = min(min_gen, gen)
    #             break

    #     if(gen == max_iter):
    #         gen_not_conv += 1
    #     else:
    #         gen_conv.append(state_with_max_fitness)
    #     print('Epoch: ', epoch, end='\t')
    #     print("Generation: ", gen, end='\t')
    #     print("Max Fitness: ", max_fitness,
    #           "state with max fitness: ", state_with_max_fitness)

    # print("For ", num_epochs, " epochs: ")
    # print("Number of epochs without convergence: ", gen_not_conv)
    # print("Avg number of generations for convergernce: ",
    #       gen_sum/(num_epochs-gen_not_conv))

    # print("Min convergence generation: ", min_gen)
    # print("Max convergence generation: ", max_gen)

    # print('\n\n------------------------------------------ IMPROVED VERSION -------------------------------------------------\n\n')

    # gen_sum = 0
    # gen_not_conv = 0
    # max_gen = 0
    # min_gen = 10000
    # gen_conv = []
    # for epoch in range(num_epochs):
    #     state = "44444444"
    #     population = []
    #     for i in range(20):
    #         population.append(state)
    #     gen = max_iter
    #     for itr in range(max_iter):
    #         population = genetic_algo_improved(population)
    #         fitness = calc_fitness(population)
    #         max_fitness = 0
    #         state_with_max_fitness = []
    #         for i in range(20):
    #             if(fitness[i] > max_fitness):
    #                 max_fitness = fitness[i]
    #                 state_with_max_fitness = population[i]

    #         # print("Max Fitness: ", max_fitness,
    #         #       "state with max fitness: ", state_with_max_fitness)

    #         if(max_fitness == 29):
    #             gen = itr+1
    #             gen_sum += gen
    #             max_gen = max(max_gen, gen)
    #             min_gen = min(min_gen, gen)
    #             break

    #     if(gen == max_iter):
    #         gen_not_conv += 1
    #     else:
    #         gen_conv.append(state_with_max_fitness)
    #     print('Epoch: ', epoch, end='\t')
    #     print("Generation: ", gen, end='\t')
    #     print("Max Fitness: ", max_fitness,
    #           "state with max fitness: ", state_with_max_fitness)

    # print("For ", num_epochs, " epochs: ")
    # print("Number of epochs without convergence: ", gen_not_conv)
    # print("Avg number of generations for convergernce: ",
    #       gen_sum/(num_epochs-gen_not_conv))

    # print("Min convergence generation: ", min_gen)
    # print("Max convergence generation: ", max_gen)

    # Uncomment till this part for average performance


if __name__ == '__main__':
    main()
