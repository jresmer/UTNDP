from random import choice
from consts import Consts
from networkx import DiGraph, all_simple_paths
from bus_line import BusLine


class Utils:

    """
    TODO
    rever alocação de frota
    primeiro botar um ônibus em cada rota e dps usar a proporcionalidade
    """
    @staticmethod
    def allocate_fleet(routeset: list, demand_matrix: list, t_fleet: int):

        for route in routeset:

            route.reset_fleet()

        route_demand = [0] * len(routeset)
        t_demand = 0

        for i, route in enumerate(routeset):

            stops = route.get_stops()

            for j in range(len(stops) - 1):

                edge_demand = demand_matrix[stops[j]][stops[j+1]]
                route_demand[i] += edge_demand
                t_demand += edge_demand

        for route in routeset:

            route.add_buses(1)
            t_fleet -= 1

        demand_per_vehicle = t_demand / t_fleet
        max_unsatisfied_demand = 0
        unused_fleet_candidate = None

        for i, route in enumerate(routeset):

            allocated_fleet = route_demand[i] // demand_per_vehicle
            if allocated_fleet > t_fleet: allocated_fleet = t_fleet
            route.add_buses(allocated_fleet)
            t_fleet -= allocated_fleet
            unsatisfied_demand = route_demand[i] % demand_per_vehicle

            if max_unsatisfied_demand < unsatisfied_demand:

                max_unsatisfied_demand = unsatisfied_demand
                unused_fleet_candidate = i

        if t_fleet > 0:

            if unused_fleet_candidate is None:

                i = 0

                while t_fleet:

                    routeset[i].add_buses(1)
                    t_fleet -= 1
                    i += 1
            else:

                routeset[unused_fleet_candidate].add_buses(t_fleet)

    @staticmethod
    def repair(g: DiGraph, routeset: list, unreached_stops: set, max_vertices: int):

            def add_stop(g: DiGraph, route: BusLine, pos: int, unreached_stops: set):

                if pos == 0:

                    possible_stops = set(g.predecessors(route.at(pos)))
                    possible_stops = list(possible_stops.intersection(unreached_stops))
                    if possible_stops:

                        new_stop = choice(possible_stops)

                        route.add_stop(
                            position=pos,
                            stop=new_stop
                        )

                        unreached_stops.remove(new_stop)

                        return_route = Utils.recalculate_return_route(g, route.at(-1), route.at(0))
                        route.set_return_route(return_route)

                elif pos < route.get_length():

                    possible_stops = set(g.predecessors(route.at(pos)))
                    possible_stops = possible_stops.intersection(set(g.successors(route.at(pos - 1))))
                    possible_stops = list(possible_stops.intersection(unreached_stops))

                    if possible_stops:

                        new_stop = choice(possible_stops)

                        route.add_stop(
                            position=pos,
                            stop=new_stop,
                        )

                        unreached_stops.remove(new_stop)

                else:

                    possible_stops = set(g.neighbors(route.at(-1)))
                    possible_stops = list(possible_stops.intersection(unreached_stops))

                    if possible_stops:

                        new_stop = choice(possible_stops)

                        route.add_stop(
                            position=pos,
                            stop=new_stop,
                        )

                        unreached_stops.remove(new_stop)

                        return_route = Utils.recalculate_return_route(g, route.at(-1), route.at(0))
                        route.set_return_route(return_route)
        
            routes = list(range(len(routeset)))

            while routes:

                route = choice(routes)
                routes.remove(route)
                route = routeset[route]

                # Adds up to 2 unreached stops to the route if possible respecting maximum route length and its stops neighborhoods
                if route.get_length() < max_vertices:

                    for _ in range(g.order() * 2):

                        if not unreached_stops or route.get_length() == max_vertices:

                            break

                        for i in range(route.get_length()):

                            add_stop(
                                g=g,
                                route=route,
                                pos=i,
                                unreached_stops=unreached_stops
                            )

            return routeset
    
    @staticmethod
    def recalculate_return_route(g: DiGraph, stop0: int, stopn: int, method: str="dfs") -> list:


        if method == "dfs":

            frontier = list(g.successors(stop0))
            a = {vertice: stop0 for vertice in frontier}
            a[stop0] = None
            next_stop = None

            while next_stop != stopn:

                if not frontier:

                    return []

                next_stop = frontier.pop(0)

                sucessors = list(g.successors(next_stop))
                for stop in sucessors:

                    if stop not in a.keys():

                        a[stop] = next_stop
                        frontier.insert(0, stop)

            return_route = [next_stop]
            p = a[next_stop]

            while p != None:

                return_route.append(p)
                p = a[p]

            return_route.reverse()
            
            return return_route
        
        elif method == "randomchoice":

            paths = list(all_simple_paths(g, stop0, stopn))
            if len(paths) < 1:

                return []
            else:
                return choice(paths)
