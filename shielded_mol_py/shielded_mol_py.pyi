
class SystemParams:
    l_max: int
    parity: str
    c6: float
    c3: float
    ksi: float
    trap_freq: float
    mass: float
    energy: float

    def __init__(
        self, 
        l_max: int,
        parity: str,
        c6: float,
        c3: float,
        ksi: float,
        trap_freq: float,
        mass: float,
        energy: float,
    ) -> None: ...

class BoundState:
    energy: float
    nodes: int

class Problem:
    def __init__(
        self, 
        params: 
        SystemParams, 
        r_min: float, 
        r_match: float, 
        r_max:float
    ) -> None: ...

    def step_rule(self, dr_min: float, dr_max: float, wavelength_ratio: float) -> None: ...

    def node_range(self, node_min: int, node_max: int) -> None: ...

    def bound_states(self, e_range: tuple[float, float], e_err: float) -> list[BoundState]: ...
    
    def wave_function(self, bound_state: BoundState) -> tuple[list[float], list[list[float]]]: ...

    def channels(self) -> list[tuple[int, int]]: ...
