from math import ceil


# TODO - REVISAR

class BusLine:

    def __init__(self, id: int, starting_point: int, fleet: int=1):

        self._id = id
        self._fleet = fleet
        self._q0 = starting_point
        self._stops = [starting_point]
        self._return_route = []

    def add_stop(self, position: int, stop: int):

        self._stops.insert(position, stop)

    def remove_stop(self, position: int): # remover o tempo tomado pela parada
        
        self._stops.pop(position)

    def substitute_stop(self, position: int, stops: list): # deve ser possível??

        self._stops = self._stops[:position] + stops + self._stops[position+1:]

    def add_buses(self, extra_fleet: int=1):

        self._fleet += extra_fleet

    def reset_fleet(self):

        self._fleet = 0

    def at(self, position: int) -> int:

        if position < len(self._stops):

            return self._stops[position]
        
        else:

            position -= len(self._stops)

            if position < len(self._return_route):

                return self._return_route[position]

            else:

                return None

    def get_stops(self) -> list:

        return self._stops + self._return_route[1:-1]
    
    def get_main_route(self) -> list:

        return self._stops
    
    def get_return_route(self) -> list:

        return self._return_route
    
    def get_fleet(self) -> int:

        return self._fleet
    
    def get_id(self) -> int:

        return self._id
    
    def get_length(self) -> int:

        return len(self._stops)
    
    def __eq__(self, other: list) -> bool:

        return self._stops + self._return_route[1:-1] == other.get_stops()
    
    def set_return_route(self, return_route: tuple) -> bool:

        if self._stops[0] == self._stops[-1]:

            self._return_route = return_route
            return True
    
        if return_route[-1] == self._stops[0] and \
            return_route[0] == self._stops[-1]:

            self._return_route = return_route

            return True
        
        return False

    def set_main_route(self, main_route: tuple) -> None:

        self._stops = main_route
        self._return_route = []
