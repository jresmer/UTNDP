from math import ceil


class BusLine:

    def __init__(self, starting_point: (int, int), turnaround_time: float, fleet: int=1):

        self._fleet = fleet
        self._q0 = starting_point
        self._stops = list()
        self._turnaround_time = turnaround_time
        self._frequency = ceil(turnaround_time / fleet)
        self.__standard_frequency = True

    def __update_frequency(self):

        self._frequency = ceil(self._turnaround_time / self._fleet)

    def add_stop(self, position: int, stop: int, added_time: float):

        self._stops.insert(position, stop)

        self._turnaround_time += added_time

    def remove_stop(self, position: int): # remover o tempo tomado pela parada

        self.__stops.pop(position)

    def substitute_stop(self, position: int, stop: int): # deve ser possível??

        self._stops[position] = stop

    def add_bus(self):

        self._fleet += 1

        # updates frequency
        if self.__standard_frequency:

            self.__update_frequency()

    def get_stops(self) -> list:

        return self._stops
    
    def get_fleet(self) -> int:

        return self._fleet

    