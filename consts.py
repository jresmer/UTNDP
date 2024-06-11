
class Consts:

    def __init__(self,
                 tp: int=20,
                 fs: int=15,
                 min_v: int=4,
                 max_v: int=8,
                 min_l: int=6,
                 usp: int=120,
                 mp: float=0.1,
                 ps: int=200) -> None:
        
        self._indexes = [0, 0, 0, 0, 0]
        self._i = 4

        self.__TRANFER_PENALTY = tp
        self.__FLEET_SIZE = fs
        self.__MIN_VERTICES = min_v
        self.__MAX_VERTICES = max_v
        self.__MIN_LINES = min_l
        self.__UNREACHABLE_STOP_PENALTY = usp
        self.__MUTATION_PROBABILITY = mp
        self.__POPULATION_SIZE = ps

    @property
    def TRANFER_PENALTY(self):

        return self.__TRANFER_PENALTY
    
    @property
    def FLEET_SIZE(self):

        return self.__FLEET_SIZE
    
    @property
    def MIN_VERTICES(self):

        return self.__MIN_VERTICES
    
    @property
    def MIN_VERTICES(self):

        return self.__MIN_VERTICES

    @property
    def MAX_VERTICES(self):

        return self.__MAX_VERTICES
    
    @property
    def MIN_LINES(self):

        return self.__MIN_LINES
    
    @property
    def UNREACHABLE_STOP_PENALTY(self):

        return self.__UNREACHABLE_STOP_PENALTY
    
    @property
    def MUTATION_PROBABILITY(self):

        return self.__MUTATION_PROBABILITY
    
    @property
    def POPULATION_SIZE(self):

        return self.__POPULATION_SIZE
    
    def _set_tp(self, v: int):
        
        self.__TRANFER_PENALTY = v
    
    def _set_min_v(self, v: int):
        
        self.__MIN_VERTICES = v

    def _set_max_v(self, v: int):
        
        self.__MAX_VERTICES = v

    def _set_mp(self, v: float):
        
        self.__MUTATION_PROBABILITY = v

    def _set_usp(self, v: int):
        
        self.__UNREACHABLE_STOP_PENALTY = v

    def set_variable_params(self,
                           tp: int=20,
                           min_v: int=4,
                           max_v: int=8,
                           usp: int=120,
                           mp: float=0.1) -> None:
        
        self.__TRANFER_PENALTY = tp
        self.__MIN_VERTICES = min_v
        self.__MAX_VERTICES = max_v
        self.__UNREACHABLE_STOP_PENALTY = usp
        self.__MUTATION_PROBABILITY = mp

    def reset(self) -> None:

        for i in range(len(self._par)):
            
            self._t[i](self._par[i][0])
