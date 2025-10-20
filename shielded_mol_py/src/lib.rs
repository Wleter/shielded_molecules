use pyo3::prelude::*;
use shielded_molecules::{
    self as sm,
    cc_problems::{self, hilbert_space::Parity, prelude::*}
};

#[pyclass]
struct SystemParams {
    #[pyo3(get, set)]
    l_max: u32,
    #[pyo3(get, set)]
    m_parity: String,
    #[pyo3(get, set)]
    symmetry: String,
    #[pyo3(get, set)]
    entrance: (u32, i32),

    #[pyo3(get, set)]
    c6: f64,
    #[pyo3(get, set)]
    c3: f64,
    #[pyo3(get, set)]
    ksi: f64,

    #[pyo3(get, set)]
    trap_freq: f64,
    #[pyo3(get, set)]
    mass: f64,
    #[pyo3(get, set)]
    energy: f64,
}

#[pymethods]
impl SystemParams {
    #[new]
    fn new(
        l_max: u32,
        m_parity: String,
        symmetry: String,
        entrance: (u32, i32),
        c6: f64,
        c3: f64,
        ksi: f64,
        trap_freq: f64,
        mass: f64,
        energy: f64,
    ) -> Self {
        Self {
            l_max,
            m_parity,
            symmetry,
            entrance,
            c6,
            c3,
            ksi,
            trap_freq,
            mass,
            energy,
        }
    }
}

#[pyclass]
struct BoundState {
    #[pyo3(get)]
    energy: f64,

    #[pyo3(get)]
    nodes: u64,
}

#[pymethods]
impl BoundState {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("energy: {}, nodes: {}", self.energy, self.nodes))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "<BoundState energy={} nodes={}>",
            self.energy, self.nodes
        ))
    }
}

#[pyclass]
struct SMatrix {
    #[pyo3(get)]
    elastic_cross_sect: f64,

    #[pyo3(get)]
    scattering_length_re: f64,

    #[pyo3(get)]
    scattering_length_im: f64,

    #[pyo3(get)]
    inelastic_cross_sections: Vec<f64>,
}

#[pyclass]
struct Problem {
    problem: sm::SystemProblem,
    r_range: (f64, f64, f64),
    step_rule: LocalWavelengthStep,
    node_range: Option<[u64; 2]>,
}

#[pymethods]
impl Problem {
    #[new]
    fn new(params: &SystemParams, r_min: f64, r_match: f64, r_max: f64) -> Self {
        let parity = match params.m_parity.as_str() {
            "all" => Parity::All,
            "even" => Parity::Even,
            "odd" => Parity::Odd,
            _ => panic!("Only all/even/odd values are allowed"),
        };
        let symmetry = match params.symmetry.as_str() {
            "none" => Parity::All,
            "fermionic" => Parity::Odd,
            "bosonic" => Parity::Even,
            _ => panic!("Only none/fermionic/bosonic values are allowed"),
        };

        let params = sm::SystemParams {
            l_max: params.l_max,
            parity,
            symmetry,
            entrance: params.entrance,
            c6: params.c6,
            c3: params.c3,
            ksi: params.ksi,
            trap_freq: params.trap_freq,
            mass: params.mass,
            energy: params.energy,
        };

        Problem {
            problem: sm::get_problem(&params),
            r_range: (r_min, r_match, r_max),
            step_rule: LocalWavelengthStep::default(),
            node_range: None,
        }
    }

    pub fn channels(&self) -> Vec<(u32, i32)> {
        self.problem.channels.clone()
    }

    fn step_rule(&mut self, dr_min: f64, dr_max: f64, wavelength_ratio: f64) {
        self.step_rule = LocalWavelengthStep::new(dr_min, dr_max, wavelength_ratio)
    }

    fn node_range(&mut self, node_min: u64, node_max: u64) {
        self.node_range = Some([node_min, node_max])
    }

    pub fn bound_states(&self, e_range: (f64, f64), e_err: f64) -> Vec<BoundState> {
        let problem = &self.problem.red_coupling;

        let mut bound_finder = BoundStatesFinder::default()
            .set_problem(|e| {
                let mut problem = problem.clone();
                problem.asymptote.set_energy(e * AuEnergy);

                problem
            })
            .set_parameter_range(e_range.into(), e_err)
            .set_propagator(|b, w| JohnsonLogDerivative::new(w, self.step_rule.into(), b))
            .set_r_range([self.r_range.0 * Bohr, self.r_range.1 * Bohr, self.r_range.2 * Bohr]);

        if let Some(node_range) = self.node_range {
            bound_finder = bound_finder.set_node_range(NodeRangeTarget::Range(node_range[0], node_range[1]));
        }

        let bound_states = bound_finder.bound_states();

        bound_states
            .map(|b| {
                let b = b.unwrap();

                BoundState {
                    energy: b.parameter,
                    nodes: b.node,
                }
            })  
            .collect()
    }

    pub fn scattering(&self, r_min: f64, r_max: f64) -> SMatrix {
        let boundary = vanishing_boundary(r_min * Bohr, Direction::Outwards, &self.problem.red_coupling);

        let mut propagator = RatioNumerov::new(&self.problem.red_coupling, self.step_rule.into(), boundary);
        let solution = propagator.propagate_to(r_max);

        let s_matrix = solution.get_s_matrix(&self.problem.red_coupling);

        SMatrix {
            elastic_cross_sect: s_matrix.get_elastic_cross_sect(),
            scattering_length_re: s_matrix.get_scattering_length().re,
            scattering_length_im: s_matrix.get_scattering_length().im,
            inelastic_cross_sections: s_matrix.get_inelastic_cross_sects(),
        }
    }

    pub fn wave_function(&self, bound_state: &BoundState) -> (Vec<f64>, Vec<Vec<f64>>) {
        let problem = &self.problem.red_coupling;
        let bound_finder = BoundStatesFinder::default()
            .set_problem(|e| {
                let mut problem = problem.clone();
                problem.asymptote.set_energy(e * AuEnergy);

                problem
            })
            .set_propagator(|b, w| JohnsonLogDerivative::new(w, self.step_rule.into(), b))
            .set_r_range([self.r_range.0 * Bohr, self.r_range.1 * Bohr, self.r_range.2 * Bohr]);

        let wave_function = bound_finder
            .bound_wave(&cc_problems::bound_states::BoundState { 
                parameter: bound_state.energy, 
                node: bound_state.nodes, 
                occupations: None
            });

        (wave_function.distances, wave_function.values)
    }
}

#[pymodule]
fn shielded_mol_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SystemParams>()?;
    m.add_class::<BoundState>()?;
    m.add_class::<SMatrix>()?;
    m.add_class::<Problem>()?;

    Ok(())
}
