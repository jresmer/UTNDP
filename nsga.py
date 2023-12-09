import networkx as nx

class SGA:

    # operator cost function
    def operator_cost(self, g: nx.DiGraph, r: set) -> float:
        """
        The operator cost for a routset r is the sum of the costs of all routes
        The cost of a route i is given by the formula: available fleet by route i / nodes reached by rout i
        """

        cost = 0

        for route in r:

            cost += route.get_fleet() / len(route.get_stops())

        return cost
    
    # passenger cost
    def passanger_cost(self, g: nx.DiGraph, r: set) -> float:
        """
        The passanger cost for a routset r is the average time a passanger takes to travel from a node u to a node v
        by using the routes available added to a possible penalty in case a certain destiny v is not reachable from a starting pont u
        """

        def floyd_warshall(g: nx.DiGraph, edges: dict): # modificar para identificar os caminhos percorridos
            """
            Variation of the FLoyd-Warshall algorithm taking into acount how the bus-routes were set
            """

            n = g.number_of_nodes()
            distancies = [None] * n
            nodes = [node for node in g.nodes]

            distancies[0] = [[0 if i == j else float('inf') if not g.has_edge(i, j) or edges[(i,j)] is None else g[i,j]["weight"] for j in nodes] for i in nodes]
            a = [[i if g.has_edge(i, j) else None for j in nodes] for i in nodes]

            for k in range(1, n):

                distancies[k] = [row.copy() for row in distancies[k-1]]

                for i in range(n):
                    
                    for j in range(n):
                        
                        if distancies[k-1][i][j] > distancies[k-1][i][k] + distancies[k-1][k][j]:

                            distancies[k][i][j] = distancies[k-1][i][k] + distancies[k-1][k][j]
                            a[i][j] = k

            return distancies[-1], a
        
        edges = {e: None for e in g.edges}

        for route in r:

            stops = route.get_stops()

            for i in range(len(stops) - 1):

                if not edges[(stops[i], stops[i+1])]:

                    edges[(stops[i], stops[i+1])] = [route.get_id()]
                else:

                    edges[(stops[i], stops[i+1])].append(route.get_id())

        d, a = floyd_warshall(g, edges)

        # TODO - algoritmo identifica as transferencias e soma as penalidades