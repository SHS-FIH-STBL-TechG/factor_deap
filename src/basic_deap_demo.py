# src/basic_deap_demo.py
import random
from deap import base, creator, tools, algorithms


# 1. 定义“最大化适应度”和“个体”类型
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

GENE_LENGTH = 50  # 个体长度：50 位 0/1

# 2. 定义基因、个体、种群
toolbox.register("attr_bit", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bit, n=GENE_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. 适应度函数：Onemax（1 越多越好）
def eval_one_max(individual):
    return sum(individual),  # 注意最后有个逗号，表示返回一个元组

toolbox.register("evaluate", eval_one_max)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    pop = toolbox.population(n=100)
    NGEN = 20

    for gen in range(NGEN):
        # 交叉 + 变异
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)

        # 评估适应度
        fits = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        # 选择下一代
        pop = toolbox.select(offspring, k=len(pop))

        best = tools.selBest(pop, k=1)[0]
        print(f"Gen {gen:02d}, best fitness = {best.fitness.values[0]}")

    print("Final best individual:", tools.selBest(pop, k=1)[0])


if __name__ == "__main__":
    main()
