use faer_ext::{IntoFaer, numpy::PyReadonlyArray2};
use pyo3::prelude::*;
use cc_problems::{coupled_chan::coupling::{Asymptote, Levels, RedCoupling}, prelude::*};

#[pyclass]
struct NumpyPotential(Py<PyAny>, usize);

impl Clone for NumpyPotential {
    fn clone(&self) -> Self {
        let cloned = pyo3::Python::attach(|py| self.0.clone_ref(py));

        Self(cloned, self.1.clone())
    }
}

#[pymethods]
impl NumpyPotential {
    #[new]
    pub fn new(size: usize, potential_func: Py<PyAny>) -> Self {
        Self(potential_func, size)
    }
}

impl VanishingCoupling for NumpyPotential {
    fn value_inplace(&self, r: f64, channels: &mut cc_problems::coupled_chan::Operator) {
        pyo3::Python::attach(|py| {
            let array: PyReadonlyArray2<f64> = self.0.call1(py, (r,))
                .expect("Expected function with signature fn(r) -> np.array")
                .extract(py)
                .expect("Expected function with signature fn(r) -> np.array");
            let array = array.into_faer();

            channels.0.copy_from(array);
        });
    }

    fn value_inplace_add(&self, r: f64, channels: &mut cc_problems::coupled_chan::Operator) {
        pyo3::Python::attach(|py| {
            let array: PyReadonlyArray2<f64> = self.0.call1(py, (r,))
                .expect("Expected function with signature fn(r) -> np.array")
                .extract(py)
                .expect("Expected function with signature fn(r) -> np.array");
            let array = array.into_faer();

            channels.0 += array;
        });
    }

    fn size(&self) -> usize {
        self.1
    }
}

// #[pyclass]
// struct PotentialCurve(Py<PyAny>);

// #[pymethods]
// impl PotentialCurve {
//     #[new]
//     pub fn new(potential_func: Py<PyAny>) -> Self {
//         Self(potential_func)
//     }
// }

// impl Interaction for NumpyPotential {
//     fn value(&self, r: f64) -> f64 {
//         let mut value = 0.;
//         pyo3::Python::attach(|py| {
//             value = self.0.call1(py, (r,))
//                 .expect("Expected callable in PotentialCurve of the form fn(float) -> float")
//                 .extract(py)
//                 .expect("Expected callable in PotentialCurve of the form fn(float) -> float")
//         });

//         value
//     }
// }

#[pyclass]
struct SystemParams {
    #[pyo3(get, set)]
    channels: Vec<(u32, i32)>,
    #[pyo3(get, set)]
    entrance: usize,
    #[pyo3(get, set)]
    potential: NumpyPotential,
    #[pyo3(get, set)]
    mass: f64,
    #[pyo3(get, set)]
    energy: f64,
}

#[pymethods]
impl SystemParams {
    #[new]
    fn new(
        channels: Vec<(u32, i32)>,
        entrance: usize,
        potential: NumpyPotential,
        mass: f64,
        energy: f64,
    ) -> Self {
        Self {
            channels,
            entrance,
            potential,
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
    red_coupling: RedCoupling<NumpyPotential>,
    r_range: (f64, f64, f64),
    step_rule: LocalWavelengthStep,
    node_range: Option<[u64; 2]>,
}

#[pymethods]
impl Problem {
    #[new]
    fn new(params: &SystemParams, r_min: f64, r_match: f64, r_max: f64) -> Self {
        let levels = Levels { 
            l: params.channels.iter().map(|x| x.0).collect(), 
            asymptote: vec![0.; params.potential.size()] 
        };

        let asymptote = Asymptote::new_diagonal(
            Quantity(params.mass, AuMass), 
            Quantity(params.energy, AuEnergy), 
            levels, 
            params.entrance
        );

        let red_coupling = RedCoupling::new(params.potential.clone(), asymptote);

        Problem {
            red_coupling,
            r_range: (r_min, r_match, r_max),
            step_rule: LocalWavelengthStep::default(),
            node_range: None,
        }
    }

    fn step_rule(&mut self, dr_min: f64, dr_max: f64, wavelength_ratio: f64) {
        self.step_rule = LocalWavelengthStep::new(dr_min, dr_max, wavelength_ratio)
    }

    fn node_range(&mut self, node_min: u64, node_max: u64) {
        self.node_range = Some([node_min, node_max])
    }

    pub fn bound_states(&self, e_range: (f64, f64), e_err: f64) -> Vec<BoundState> {
        let problem = &self.red_coupling;

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
        let boundary = vanishing_boundary(r_min * Bohr, Direction::Outwards, &self.red_coupling);

        let mut propagator = RatioNumerov::new(&self.red_coupling, self.step_rule.into(), boundary);
        let solution = propagator.propagate_to(r_max);

        let s_matrix = solution.get_s_matrix(&self.red_coupling);

        SMatrix {
            elastic_cross_sect: s_matrix.get_elastic_cross_sect(),
            scattering_length_re: s_matrix.get_scattering_length().re,
            scattering_length_im: s_matrix.get_scattering_length().im,
            inelastic_cross_sections: s_matrix.get_inelastic_cross_sects(),
        }
    }

    pub fn wave_function(&self, bound_state: &BoundState) -> (Vec<f64>, Vec<Vec<f64>>) {
        let problem = &self.red_coupling;
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
fn shielded_molecules(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SystemParams>()?;
    m.add_class::<BoundState>()?;
    m.add_class::<SMatrix>()?;
    m.add_class::<Problem>()?;
    m.add_class::<NumpyPotential>()?;

    Ok(())
}
