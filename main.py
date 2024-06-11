import sys
import networkx as nx
from time import time
from os import path
from nsga import NSGA, PopGenerator
from consts import Consts
from log_manager import LogManager
from population_dao import PopulationDAO


def prepare_instance(prefix : str) -> nx.DiGraph:

    g = nx.DiGraph()

    # lendo arquivo de coordenadas / reading coords file
    file_path = path.join("Instances", prefix + "Coords.txt")

    with open(file_path) as f:

        node = 0
        next(f)

        for coord in f:

            x, y = coord.split()
            x, y  = float(x), float(y)

            g.add_node(node, coord=(x,y))

            node += 1

    # lendo matriz de tempo de trajeto / reading travel time matrix
    file_path = path.join("Instances", prefix + "TravelTimes.txt")

    with open(file_path) as f:

        for i, row in enumerate(f):

            row = row.split()

            for j, cell in enumerate(row):

                if cell not in ("Inf", "0"):

                    g.add_edge(i, j, weight=float(cell))

    # lendo matriz de demanda / reading demand matrix
    file_path = path.join("Instances", prefix + "Demand.txt")

    demand = []

    with open(file_path) as f:

        for i, row in enumerate(f):

            demand.append(list())
            row = row.split()

            for j, cell in enumerate(row):

                demand[i].append(int(cell))

    return g, demand

def main():

    if __name__ == "__main__":

        # padrão para testes / testing mode
        if len(sys.argv) == 1:

            consts = Consts()
            log = LogManager("refactor_log.csv")
            dao = PopulationDAO("refactor_pop_data.txt")
            gen = PopGenerator()
            unsga3 = NSGA(consts)

            # writing log header
            log.write_header()

            # instances
            instances = ["Mumford0"]

            for instance in instances:

                best_cost = (float("inf"), float("inf"))
                best_pop = None

                # preparando intâncias / preparing instances
                g, d = prepare_instance(instance)

                for tp in [20, 15, 25]:

                    for min_v in [4, 3]:

                        for max_v in [8, 10]:

                            for mp in [0.1, 0.25, 0.5]:

                                for usp in [120, 200]:

                                    consts.set_variable_params(tp, min_v, max_v, usp, mp)
                                    start_time = time()

                                    pop = gen.generate_population(
                                        g=g,
                                        routeset_size=consts.MIN_LINES,
                                        population_size=200,
                                        total_fleet=consts.FLEET_SIZE,
                                        max_vertices=consts.MAX_VERTICES,
                                        min_vertices=consts.MIN_VERTICES,
                                        demand_matrix=d
                                    )

                                    pop = unsga3.loop(
                                        max_generations=200,
                                        population=pop,
                                        g=g,
                                        reference_points=[(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],
                                        demand_matrix=d
                                    )

                                    _, costs = unsga3.get_best_individual(g, d)

                                    if costs[0] <= best_cost[0] and costs[1] <= best_cost[1] and \
                                        costs[0] < best_cost[0] or costs[1] < best_cost[1]:

                                        best_cost = costs
                                        best_pop = pop
                                        dao.add(best_pop)

                                    end_time = time()

                                    log.write_row(
                                        instance=instance,
                                        population_size=consts.POPULATION_SIZE,
                                        mutation_prob=consts.MUTATION_PROBABILITY,
                                        fleet_size=consts.FLEET_SIZE,
                                        min_lines=consts.MIN_LINES,
                                        min_vertices=consts.MIN_VERTICES,
                                        max_vertices=consts.MAX_VERTICES,
                                        transfer_penalty=consts.TRANFER_PENALTY,
                                        unreached_stop_penalty=consts.UNREACHABLE_STOP_PENALTY,
                                        best_ind=costs,
                                        start_time=start_time,
                                        end_time=end_time
                                    )   

                                    consts.vary()

# roda o programa / runs program
main()
