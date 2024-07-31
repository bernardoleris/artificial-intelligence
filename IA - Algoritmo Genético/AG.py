import random
import time
import matplotlib.pyplot as plt

def binary_to_decimal(binary):
    decimal = 0
    for bit in binary:
        decimal = (decimal << 1) | bit
    if binary[0] == 1:  # Número negativo
        decimal -= (1 << len(binary))
    return decimal

def decimal_to_binary(decimal, length):
    if decimal < 0:
        decimal = (1 << length) + decimal
    binary = [int(x) for x in bin(decimal)[2:]]
    while len(binary) < length:
        binary.insert(0, 0)
    return binary

def fitness_function(x):
    return x**2 - 3*x + 4

def create_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

def create_population(size, length):
    return [create_individual(length) for _ in range(size)]

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]
    else:
        return parent1

def mutate(individual, mutation_rate):
    return [bit if random.random() > mutation_rate else 1 - bit for bit in individual]

def genetic_algorithm(pop_size, gen_count, crossover_rate, mutation_rate):
    length = len(decimal_to_binary(10, 5))  # Encontrar o comprimento necessário para representar o intervalo [-10, 10]
    population = create_population(pop_size, length)
    best_fitness_per_gen = []
    avg_fitness_per_gen = []

    start_time = time.time()  # Início da medição do tempo

    for generation in range(gen_count):
        fitnesses = [fitness_function(binary_to_decimal(individual)) for individual in population]
        best_fitness_per_gen.append(max(fitnesses))
        avg_fitness_per_gen.append(sum(fitnesses) / len(fitnesses))
        new_population = []
        for _ in range(pop_size):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2, crossover_rate)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    fitnesses = [fitness_function(binary_to_decimal(individual)) for individual in population]
    best_individual = population[fitnesses.index(max(fitnesses))]
    best_value = binary_to_decimal(best_individual)
    best_fitness = max(fitnesses)
    
    end_time = time.time()  # Fim da medição do tempo
    execution_time = end_time - start_time

    return best_individual, best_value, best_fitness, best_fitness_per_gen, avg_fitness_per_gen, execution_time

# Parâmetros do AG
pop_size = 15  # Pode ser ajustado para até 30
gen_count = 10  # Pode ser ajustado para até 20
crossover_rate = 0.7
mutation_rate = 0.01

# Definir a semente aleatória para resultados reproduzíveis
random.seed(42)

# Executar o AG
best_individual, best_value, best_fitness, best_fitness_per_gen, avg_fitness_per_gen, execution_time = genetic_algorithm(pop_size, gen_count, crossover_rate, mutation_rate)

print(f"População (pop_size): {pop_size}")
print(f"Gerações (gen_count): {gen_count}")
print(f"Melhor indivíduo: {best_individual}")
print(f"Melhor valor de x: {best_value}")
print(f"Melhor valor da função f(x): {best_fitness}")
print(f"Tempo de execução: {execution_time:.4f} segundos")

# Plotar a evolução da fitness
plt.plot(best_fitness_per_gen, label='Melhor Fitness')
plt.plot(avg_fitness_per_gen, label='Fitness Médio')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.legend()
plt.title('Evolução da Fitness ao Longo das Gerações')
plt.show()
