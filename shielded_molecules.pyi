
class NumpyPotential:
    def __init__(self, size: int, potential) -> None: ...

class SystemParams:
    levels: list[tuple[int, int]]
    entrance: int
    potential: NumpyPotential
    mass: float
    energy: float

    def __init__(
        self, 
        levels: list[tuple[int, int]],
        entrance: int,
        potential: NumpyPotential,
        mass: float,
        energy: float,
    ) -> None: ...

class BoundState:
    energy: float
    nodes: int

class SMatrix:
    elastic_cross_sect: float
    scattering_length_re: float
    scattering_length_im: float
    inelastic_cross_sections: list[float]

class Problem:
    def __init__(
        self, 
        params: SystemParams, 
        r_min: float, 
        r_match: float, 
        r_max:float
    ) -> None: ...

    def step_rule(self, dr_min: float, dr_max: float, wavelength_ratio: float) -> None: ...

    def node_range(self, node_min: int, node_max: int) -> None: ...

    def bound_states(self, e_range: tuple[float, float], e_err: float) -> list[BoundState]: ...

    def scattering(self, r_min: float, r_max: float) -> SMatrix: ...
    
    def wave_function(self, bound_state: BoundState) -> tuple[list[float], list[list[float]]]: ...

    def channels(self) -> list[tuple[int, int]]: ...
