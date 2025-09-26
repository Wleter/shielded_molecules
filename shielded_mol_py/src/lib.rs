use pyo3::prelude::*;
    use shielded_molecules::{
    self as sm,
    scattering_problems::{
        alkali_rotor_atom::ParityBlock,
        scattering_solver::{
            log_derivatives::johnson::Johnson,
            numerovs::LocalWavelengthStepRule,
            observables::bound_states::{BoundProblemBuilder, BoundStates, Monotony},
            quantum::{cast_variant, units::{Au, Energy}},
        },
    },
};

#[pyclass]
struct SystemParams {
    #[pyo3(get, set)]
    l_max: u32,
    #[pyo3(get, set)]
    parity: String,
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
        parity: String,
        c6: f64,
        c3: f64,
        ksi: f64,
        trap_freq: f64,
        mass: f64,
        energy: f64,
    ) -> Self {
        Self {
            l_max,
            parity,
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
        Ok(format!("<BoundState energy={} nodes={}>", self.energy, self.nodes))
    }
}

#[pyclass]
struct Problem {
    problem: sm::SystemProblem,
    r_range: (f64, f64, f64),
    step_rule: LocalWavelengthStepRule,
    node_range: Option<[u64; 2]>
}

#[pymethods]
impl Problem {
    #[new]
    fn new(params: &SystemParams, r_min: f64, r_match: f64, r_max: f64) -> Self {
        let parity = match params.parity.as_str() {
            "all" => ParityBlock::All,
            "even" => ParityBlock::Positive,
            "odd" => ParityBlock::Negative,
            _ => panic!("Only all/even/odd values are allowed"),
        };

        let params = sm::SystemParams {
            l_max: params.l_max,
            parity,
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
            step_rule: LocalWavelengthStepRule::default(),
            node_range: None
        }
    }

    pub fn channels(&self) -> Vec<(u32, i32)> {
        self.problem.basis.iter()
            .map(|x| {
                let l = cast_variant!(x[0], sm::Basis::Angular);

                (l.s.double_value() / 2, l.ms.double_value() / 2)
            })
            .collect()
    }

    fn step_rule(&mut self, dr_min: f64, dr_max: f64, wavelength_ratio: f64) {
        self.step_rule = LocalWavelengthStepRule::new(dr_min, dr_max, wavelength_ratio)
    }

    fn node_range(&mut self, node_min: u64, node_max: u64) {
        self.node_range = Some([node_min, node_max])
    }

    pub fn bound_states(&self, e_range: (f64, f64), e_err: f64) -> Vec<BoundState> {
        let mut bound_states_builder = BoundProblemBuilder::new(&self.problem.particles, &self.problem.potential)
            .with_propagation(self.step_rule.clone(), Johnson)
            .with_range(self.r_range.0, self.r_range.1, self.r_range.2)
            .with_monotony(Monotony::Increasing);

        if let Some(node_range) = self.node_range {
            bound_states_builder = bound_states_builder.with_nodes_range(node_range);
        }
        let bound_states = bound_states_builder.build();

        let e_range = (Energy(e_range.0, Au), Energy(e_range.1, Au));

        let bound_states = bound_states.bound_states(e_range, Energy(e_err, Au));

        bound_states
            .energies
            .iter()
            .zip(&bound_states.nodes)
            .map(|(e, n)| BoundState {
                energy: *e,
                nodes: *n,
            })
            .collect()
    }

    pub fn wave_function(&self, bound_state: &BoundState) -> (Vec<f64>, Vec<Vec<f64>>) {
        let bound_states =
            BoundProblemBuilder::new(&self.problem.particles, &self.problem.potential)
                .with_propagation(self.step_rule.clone(), Johnson)
                .with_range(self.r_range.0, self.r_range.1, self.r_range.2)
                .build();

        let wave_function = bound_states
            .bound_waves(&BoundStates {
                energies: vec![bound_state.energy],
                nodes: vec![bound_state.nodes],
                occupations: None,
            })
            .next()
            .unwrap();

        (wave_function.distances, wave_function.values)
    }
}

#[pymodule]
fn shielded_mol_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SystemParams>()?;
    m.add_class::<BoundState>()?;
    m.add_class::<Problem>()?;

    Ok(())
}
