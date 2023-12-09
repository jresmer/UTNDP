import sys
import networkx as nx
from os import path


def prepare_instance(prefix : str) -> nx.DiGraph:

    g = nx.DiGraph()

    # lendo arquivo de coordenadas / reading coords file
    file_path = path.join("Instances", prefix + "Coords.txt")

    with open(file_path) as f:

        node = 0
        next(f)

        for coord in f:

            x, y = coord.split()
            x, y  = int(x), int(y)

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

            # preparando intâncias / preparing instances
            g, d = prepare_instance("Mumford0")

# roda o programa / runs program
main()
