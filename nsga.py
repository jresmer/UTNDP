import networkx as nx
import numpy as np
from consts import Consts
from bus_line import BusLine
from random import randint, choice, random, sample
from copy import deepcopy
from utils import Utils


class PopGenerator:

    def generate_population(self,
                            g: nx.DiGraph,
                            routeset_size: int,
                            population_size: int,
                            total_fleet: int,
                            max_vertices: int,
                            min_vertices: int,
                            demand_matrix: int
                            ):

        population = list()

        while len(population) < population_size:

            routeset = list()
            reached_stops = set()
            routeset_target_length = randint(routeset_size, int(routeset_size * 1.15))

            while len(routeset) < routeset_target_length:

                first_stop = choice(list(g.nodes))
                l = BusLine(
                    id=len(routeset),
                    starting_point=first_stop,
                    fleet=0
                )

                l_length = randint(min_vertices, max_vertices)

                next_stop = l.at(0)

                reversed_ = False
                selected_stops = {first_stop}

                while l.get_length() < l_length:

                    attempted_stops = set()
                    last_stop = next_stop
                    next_stop = None
                    possible_choices = list(set(g.neighbors(last_stop)) - selected_stops)
                    if possible_choices:

                        next_stop = choice(possible_choices)
                
                    if next_stop is None:

                        if  len(set(possible_choices) - attempted_stops) == 0:

                            break

                        if not reversed_:

                            reversed_ = True

                            route = list(reversed(l.get_stops()))

                            l = BusLine(
                                id=len(population),
                                starting_point=route[0],
                                fleet=0
                            )

                            for i in range(1, len(route)):

                                l.add_stop(
                                    position=l.get_length(),
                                    stop=route[i]
                                )

                            next_stop = l.at(l.get_length() - 1)
                            continue

                        else:

                            if l.get_length() - 1 > 0:
                                
                                attempted_stops.add(l.at(-1))
                                selected_stops.remove(l.at(-1))
                                l.remove_stop(l.get_length() - 1)
                                next_stop = l.at(l.get_length() - 1)
                                continue

                            else: 
                                break
                    
                    selected_stops.add(next_stop)
                    l.add_stop(
                        stop=next_stop,
                        position=l_length
                    )

                stops = l.get_stops()
                return_route = Utils.recalculate_return_route(g, stops[-1], stops[0])
                if len(return_route) == 0 and stops[0] != stops[-1]:
                    
                    return_route_found = False
                    while l.get_length() > self._consts.MIN_VERTICES:
                        
                        l.remove_stop(l.get_length() - 1)
                        return_route = Utils.recalculate_return_route(g, stops[-1], stops[0])

                        if len(return_route) > 0 or stops[-1] == stops[0]:
                            
                            return_route_found = True
                            break

                    if not return_route_found:

                        continue
                
                l.set_return_route(return_route)

                routeset.append(l)
                reached_stops = reached_stops.union(set(l.get_stops()))

            if len(reached_stops) < g.order():

                routeset = Utils.repair(
                    g,
                    routeset,
                    set(g.nodes) - set(reached_stops),
                    max_vertices
                    )
                
                touched_stops = set()
                
                for route in routeset:

                    touched_stops = touched_stops.union(set(route.get_stops()))

                unreacheble_stops = set(g.nodes) - touched_stops

                if len(unreacheble_stops) > 0:

                    continue
         
            population.append(routeset)
            Utils.allocate_fleet(
                routeset=routeset,
                demand_matrix=demand_matrix,
                t_fleet=total_fleet
            )
    	
        return population


class NSGA:

    def __init__(self, consts: Consts=Consts()):

        self.__last_population = None
        self._consts = consts

    # operator cost function
    def operator_cost(self, r: list) -> float:
        """
        The operator cost for a routset r is the sum of the costs of all routes
        The cost of a route i is given by the formula: nodes reached by route i / available fleet for route i
        """

        cost = 0

        for route in r:

            cost += len(route.get_stops()) / route.get_fleet()

        return cost
    
    # passenger cost
    def passanger_cost(self, g: nx.DiGraph, r: list) -> float:
        """
        The passanger cost for a routset r is the average time a passanger takes to travel from a node u to a node v
        by using the routes available added to a possible penalty in case a certain destiny v is not reachable from a starting pont u
        """

        def floyd_warshall(g: nx.DiGraph, edges: dict, max_transfers: int=len(g.nodes)): # modificar para identificar os caminhos percorridos
            """
            Variation of the FLoyd-Warshall algorithm taking into acount how the bus-routes were set and transfer costs
            """

            n = g.number_of_nodes()
            distancies = [None] * n
            nodes = [node for node in g.nodes]

            distancies[0] = [[0 if i == j else float('inf') if not g.has_edge(i,j) or edges[(i,j)] is None else g.edges[i,j]["weight"] for j in nodes] for i in nodes]
            a = [[i if g.has_edge(i,j) and edges[(i, j)] is not None else None for j in nodes] for i in nodes]
            n_tranfers = [[0 for j in nodes] for i in nodes]
            last_routes = [[edges[(i, j)] if g.has_edge(i,j) else None for j in nodes] for i in nodes]
            first_edges = [[(i, j) if g.has_edge(i,j) and edges[(i, j)] is not None else None for j in nodes] for i in nodes]

            for k in range(1, n):

                distancies[k] = [row.copy() for row in distancies[k-1]]

                for i in range(n):
                    
                    for j in range(n):

                        if i == j: 
                            
                            continue

                        new_dist = distancies[k-1][i][k] + distancies[k-1][k][j]

                        if new_dist == float("inf"):

                            continue

                        # identifies necessary route transfer
                        first_edge = first_edges[k][j]
                        if a[i][k] is None or first_edge is None: # confirmar

                            continue
                        
                        if edges[first_edge] is None or last_routes[i][a[i][k]] is None:

                            possible_routes = []
                        else:
                            possible_routes = last_routes[i][a[i][k]].intersection(edges[first_edge])

                        if not possible_routes:

                            new_dist += self._consts.TRANFER_PENALTY
                            possible_routes = edges[first_edge]
                            
                            if n_tranfers[i][k] + 1 > max_transfers: continue

                            n_tranfers[i][j] = n_tranfers[i][k] + 1

                        if distancies[k-1][i][j] > new_dist:

                            distancies[k][i][j] = new_dist
                            a[i][j] = k
                            last_routes[i][j] = possible_routes
                            first_edges[i][j] = first_edges[i][k]

            return distancies[-1], a
        
        edges = {e: None for e in g.edges}

        for route in r:

            stops = route.get_stops()

            for i in range(len(stops) - 1):

                if edges[(stops[i], stops[i+1])] is None:

                    edges[(stops[i], stops[i+1])] = set([route.get_id()])
                else:

                    edges[(stops[i], stops[i+1])].add(route.get_id())

        d, a = floyd_warshall(g, edges)

        n = g.number_of_nodes()
        total_time = 0

        for i in range(n):

            for j in range(n):

                if d[i][j] == float('inf'):

                    total_time += self._consts.UNREACHABLE_STOP_PENALTY
                else:
                    
                    total_time += d[i][j]

        if n > 0:
            return total_time / (n ** 2)
    
        return float('inf')

    # TODO - conferir comportamento com o get_stops atual
    def crossover(self, first_parent: list, second_parent: list):

        exhausted_parents = set()

        offspring = list()

        parents = [deepcopy(first_parent),
                   deepcopy(second_parent)]

        seed = choice(first_parent)
        parents[0].remove(seed)
        offspring.append(seed)
        seed = set(seed.get_stops())
        second_parent = True

        while len(offspring) < len(first_parent):

            if second_parent:

                current_parent = 1
            else:

                current_parent = 0

            possible_choices = list()
            intersections = list()

            for route in parents[current_parent]:

                if route not in offspring:

                    intersection = seed.intersection(set(route.get_stops()))

                    if intersection:
                        
                        possible_choices.append(route)
                        intersections.append(intersection)

            next_route = None
            smallest_intersection = float("inf")

            if not possible_choices: 

                if current_parent not in exhausted_parents:

                    exhausted_parents.add(current_parent)
                elif len(exhausted_parents) == 2:

                    break

                second_parent = not second_parent
                continue

            for i, route in enumerate(possible_choices):

                if route.get_length() < smallest_intersection:
                    
                    smallest_intersection = route.get_length()
                    next_route = route
            
            parents[current_parent].remove(next_route)
            offspring.append(next_route)

            seed = seed.union(set(next_route.get_stops()))
            exhausted_parents = set()
            second_parent = not second_parent

        if len(offspring) != len(first_parent):

            return False
        
        return offspring

    def mutation(self,g: nx.DiGraph, routeset: list):

        n_changes = randint(1, len(routeset))
        made_changes = 0
        routeset_ = deepcopy(routeset)

        while made_changes < n_changes:

            if not routeset_:

                break

            choice_ = random()
            chosen_route = choice(routeset_)
            reached_stops = set(chosen_route.get_stops())
            old_main_route = deepcopy(chosen_route.get_main_route())
            old_return_route = deepcopy(chosen_route.get_return_route())

            # remove stop
            if choice_ < 0.33:

                routeset_.remove(chosen_route)

                if chosen_route.get_length() == self._consts.MIN_VERTICES:

                    continue

                chosen_stop_index = choice([-1, 0])

                chosen_route.remove_stop(chosen_stop_index)

                made_changes += 1
            # add stop
            elif choice_ < 0.67:

                routeset_.remove(chosen_route)

                if chosen_route.get_length() == self._consts.MAX_VERTICES:

                    continue

                chosen_position = choice([-1, 0])
                if chosen_position == -1:
                    available_stops = g.successors(chosen_route.at(chosen_position))
                    available_stops = list(set(available_stops) - reached_stops)
                else:
                    available_stops = g.predecessors(chosen_route.at(chosen_position))
                    available_stops = list(set(available_stops) - reached_stops)
                    

                if not available_stops: continue
                new_stop = choice(available_stops)
                
                chosen_route.add_stop(
                    position=chosen_position,
                    stop=new_stop
                )

                made_changes += 1
            # substitute stop
            else:

                chosen_stop_index = choice([-1, 0])
                
                if chosen_stop_index == -1:

                    predecessor = chosen_route.at(chosen_route.get_length() - 2)
                    available_stops = g.successors(predecessor)
                    available_stops = list(set(available_stops) - reached_stops)

                    if not available_stops: continue
                    new_stop = choice(available_stops)

                else:

                    sucessor = chosen_route.at(1)
                    available_stops = g.predecessors(sucessor)
                    available_stops = list(set(available_stops) - reached_stops)
                    
                    if not available_stops: continue
                    new_stop = choice(available_stops)

                chosen_route.substitute_stop(
                    position=chosen_stop_index,
                    stops=[new_stop]
                )

                made_changes += 1

        
            stops = chosen_route.get_main_route()
            return_route = Utils.recalculate_return_route(g, stops[-1], stops[0])
            if len(return_route) > 0 or stops[-1] == stops[0]:
                chosen_route.set_return_route(return_route)
            else:
                chosen_route.set_main_route(old_main_route)
                chosen_route.set_return_route(old_return_route)
                routeset_.append(chosen_route)
                made_changes -= 1

    # Checks if a solution dominates another (2 objective functions)
    @staticmethod
    def dominates(obj_values1: float, obj_values2: float):
        """
        A more general solution would be:
        return all(ov1 <= ov2 for ov1, ov2 in zip(obj_values1, obj_values2))
        """

        ov11, ov12 = obj_values1
        ov21, ov22 = obj_values2

        return ov11 <= ov21 and ov12 <= ov22 and (ov11 < ov21 or ov12 < ov22)
    
    def non_dominanted_sort(self, obj_values: list):

        population_size = len(obj_values)
        dominance_matrix = [[0] * population_size for _ in range(population_size)]
        ranks = [0] * population_size
        ranks_ = [0] * population_size
        fronts = [[]]

        for i in range(population_size):

            for j in range(i + 1, population_size):
                
                if self.dominates(obj_values[i], obj_values[j]):

                    dominance_matrix[i][j] = 1

                elif self.dominates(obj_values[j], obj_values[i]):

                    dominance_matrix[j][i] = 1

        for i in range(population_size):

            for j in range(population_size):

                if dominance_matrix[j][i]: ranks[i] += 1

            if ranks[i] == 0: fronts[0].append(i)

        index = 0

        while len(fronts[index]) > 0:

            next_front = []

            for i in fronts[index]:
      
                for j in range(population_size):

                    if dominance_matrix[i][j] == 1: 
                        
                        ranks[j] -= 1

                        if ranks[j] == 0:

                            next_front.append(j)
                            ranks[i] = len(fronts)

            index += 1
            fronts.append(next_front)

        return fronts[:-1], ranks_
    
    def parent_selection(self, parent1, parent2, associations, ref_points, ranks, obj_values):

        pick = None
        p1_ref_point = None
        p2_ref_point = None

        for i, ref_point_associations in enumerate(associations):

            if parent1 in ref_point_associations:

                p1_ref_point = ref_points[i]

            if parent2 in ref_point_associations:

                p2_ref_point = ref_points[i]
        
        if p1_ref_point == p2_ref_point:

            if ranks[0] < ranks[1]:

                pick = parent1
            
            elif ranks[0] > ranks[1]:

                pick = parent2
            
            else:

                b, a = p1_ref_point
                x0, y0 = obj_values[parent1]
                d1 = self.calculate_distance(x0, y0, a, b)

                b, a = p2_ref_point
                x0, y0 = obj_values[parent2]
                d2 = self.calculate_distance(x0, y0, a, b)

                if d1 < d2:

                    pick = parent1

                else:

                    pick = parent2

        else:

            pick = choice((parent1, parent2))

        return pick
    
    def generate_offspring(self, g, population, associations, ranks, obj_values, demand_matrix, ref_points):

        offspring = list()
        routeset_n = []
        for niche in associations:

            routeset_n += niche

        while len(offspring) < len(population):

            possible_p1, possible_p2 = sample(routeset_n, 2)
            parent1 = self.parent_selection(
                parent1=possible_p1,
                parent2=possible_p2,
                associations=associations,
                ranks=(ranks[possible_p1], ranks[possible_p2]),
                obj_values=obj_values,
                ref_points=ref_points
            )

            possible_p1, possible_p2 = sample(routeset_n, 2)
            parent2 = self.parent_selection(
                parent1=possible_p1,
                parent2=possible_p2,
                associations=associations,
                ranks=(ranks[possible_p1], ranks[possible_p2]),
                obj_values=obj_values,
                ref_points=ref_points
            )

            child = self.crossover(population[parent1], population[parent2])

            if not child: 
                
                continue

            touched_routes = set()
            for route in child:

                touched_routes.union(set(route.get_stops()))

            if len(touched_routes) < g.order():

                child = Utils.repair(
                    g=g,
                    routeset=child,
                    unreached_stops=set(g.nodes).difference(touched_routes),
                    max_vertices=self._consts.MAX_VERTICES
                )

            if random() > self._consts.MUTATION_PROBABILITY:

                self.mutation(g, child)

            Utils.allocate_fleet(
                routeset=child,
                demand_matrix=demand_matrix,
                t_fleet=self._consts.FLEET_SIZE
            )
            
            offspring.append(child)

        return list(offspring)
    
    def calculate_distance(self, x0, y0, a, b):

        b = -b
        d = abs(a * x0 + b * y0)
        d = d / np.sqrt(np.power(a, 2) + np.power(b, 2))

        return d

    def calculate_distances(self, ref_points, individuals, obj_values):

        distances = [[0 for _ in range(len(individuals))] for _ in range(len(ref_points))]

        for individual in range(len(individuals)):

            for i, ref_point in enumerate(ref_points):

                individual_n = individuals[individual]
                x0, y0 = obj_values[individual_n]
                b, a = ref_point
                distances[i][individual] = self.calculate_distance(x0, y0, a, b)

        return distances

    def associate(self, ref_points: list, obj_values: list, selected_fronts: list):

        individuals = set()
        for front in selected_fronts:

            individuals = individuals.union(set(front))
        
        individuals = list(individuals)

        associations = [None] * len(ref_points)

        for individual in individuals:
        
            best_point = None
            shortest_dist = float("inf")
            x0, y0 = obj_values[individual]

            for i, ref_point in enumerate(ref_points):

                b, a = ref_point
                d = self.calculate_distance(x0, y0, a, b)
                if d < shortest_dist:

                    shortest_dist = d
                    best_point = i

            if associations[best_point] is None:

                associations[best_point] = list()

            associations[best_point].append(individual)

        for i in range(len(associations)):

            if associations[i] is None:

                associations[i] = list()

        return associations
            
    def niching(self, niches, associations, fronts, distance_matrix, n_individuals_to_be_selected):

        def associations_len(i: list) -> int:

            niche, associations = i
            
            return len(associations)

        niches = zip(niches, associations)
        chosen_individuals = list()
        front_n = 0
        front = fronts[front_n]

        while n_individuals_to_be_selected > 0:

            niches = sorted(niches, key=associations_len)
            if len(front) == 0:
                
                front_n += 1
                front = fronts[front_n]

            ind_selected_this_iter = min(n_individuals_to_be_selected, len(niches), len(front))
            for i in range(ind_selected_this_iter):

                sorted_front = list(enumerate(front))
                sorted_front.sort(key=lambda individual: distance_matrix[i][individual[0]])
                selected_individual = sorted_front[0][1]

                front.remove(selected_individual)
                chosen_individuals.append(selected_individual)
                associations[i].append(selected_individual)

            n_individuals_to_be_selected -= ind_selected_this_iter

        return chosen_individuals
            

    def normalize(self, obj_values, ideal_point, nadir_point):

        values = deepcopy(obj_values)

        for objective_number in range(len(obj_values[0])):

            for individual in range(len(obj_values)):

                denom = nadir_point[objective_number] - ideal_point[objective_number]
                
                if denom == 0: denom = 1e-6

                values[individual][objective_number] = (values[individual][objective_number] - ideal_point[objective_number]) / denom

        return values
    
    def loop(self, max_generations: int, population: list, g: nx.DiGraph, reference_points: list, demand_matrix: list) -> list:
    
        def routeset_union(r1, r2):

            union_r = deepcopy(r1)

            for new_routeset in r2:

                unique = True

                for old_routeset in r1:

                    if all(any(old_route == new_route for old_route in old_routeset) for new_route in new_routeset):
                            
                        unique = False
                        break
                
                if unique:

                    union_r.append(new_routeset)

            return union_r

        hyperplane = Hyperplane(2)

        offspring = list()
        associations = None

        for generation in range(max_generations):

            r = deepcopy(population)
            r = routeset_union(r, offspring)

            obj_values = list()

            for individual in r:

                individual_values = [
                     self.operator_cost(individual),
                     self.passanger_cost(g, individual)
                     ]
                
                obj_values.append(individual_values)

            fronts, ranks = self.non_dominanted_sort(obj_values)

            s = []
            i = -1
            front_ = None

            while i < len(fronts) - 1:

                i += 1
                front_ = [r[individual] for individual in fronts[i]]

                if len(s) + len(front_) > len(population):

                    break

                s = routeset_union(s, front_)

            K = len(population) - len(s)
            hyperplane.update(
                obj_values=obj_values,
                fronts=fronts
                )
            ideal_p, nadir_p = hyperplane.ideal_point, hyperplane.nadir_point
            obj_values = self.normalize(
                obj_values= obj_values, 
                ideal_point=ideal_p,
                nadir_point=nadir_p
                )
            associations = self.associate(
                ref_points=reference_points,
                obj_values=obj_values,
                selected_fronts=fronts[:i]
            )

            # last front to be included
            if K > 0:

                remainder_individuals = fronts[i]
                remainder_fronts = fronts[i:]
                for front in fronts[i+1:]:

                    remainder_individuals += front

                d = self.calculate_distances(
                    ref_points=reference_points,
                    individuals=remainder_individuals,
                    obj_values=obj_values
                )

                c = self.niching(
                    niches=reference_points,
                    fronts=remainder_fronts,
                    distance_matrix=d,
                    n_individuals_to_be_selected=K,
                    associations=associations
                )

                c = [r[individual] for individual in c]
                population = routeset_union(s, c)
                population = list(population)

            else:

                population = s
                
            offspring = self.generate_offspring(
                g=g,
                population=r,
                associations=associations,
                ranks=ranks,
                obj_values=obj_values,
                demand_matrix=demand_matrix,
                ref_points=reference_points
            )

        self.__last_population = population

        return population
    
    def get_best_individual(self, g: nx.DiGraph):

        obj_values = list()
        for individual in self.__last_population:

            individual_values = [
                 self.operator_cost(individual),
                 self.passanger_cost(g, individual)
                 ]
                
            obj_values.append(individual_values)

        fronts, _ = self.non_dominanted_sort(obj_values)

        best_ind = None
        best_ind_distance = float("inf")
        best_values = None
        for i in fronts[0]:

            individual = self.__last_population[i]

            op_cost, ps_cost = obj_values[i]
            dist = np.sqrt(np.power(op_cost, 2) + np.power(ps_cost, 2))

            if dist < best_ind_distance:

                best_ind = individual
                best_ind_distance = dist
                best_values = (op_cost, ps_cost)

        return best_ind, best_values


class Hyperplane:

    def __init__(self, n_dimensions):
        
        self.ai = [0 for _ in range(n_dimensions)]
        self.ideal_point = [float("inf") for _ in  range(n_dimensions)]
        self.nadir_point = [0 for _ in  range(n_dimensions)]

    def update(self, obj_values, fronts):

        # updates ideal values
        for i in range(len(obj_values)):

            values = obj_values[i]

            for j in range(len(values)):

                self.ideal_point[j] = min(self.ideal_point[j], values[j])

        #updates nadir values - MAXIMUM OF NON-DOMINATED FRONT (MNDF)
        current_front = 0

        while 1:

            current_values = list()

            for instance in fronts[current_front]:

                current_values.append(obj_values[instance])

            for obj_number in range(len(obj_values[0])):

                self.nadir_point[obj_number] = max([values[obj_number] for values in current_values])

            current_front += 1

            if self.stoping_condition() or current_front >= len(fronts):

                break

    def stoping_condition(self):

        return all([abs(nadir - ideal) > ideal * 0.3 for ideal, nadir in zip(self.ideal_point, self.nadir_point)]) 
