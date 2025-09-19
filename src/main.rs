use faer::Mat;
use scattering_problems::{
    alkali_rotor_atom::ParityBlock,
    scattering_solver::{
        log_derivatives::johnson::Johnson, numerovs::LocalWavelengthStepRule, observables::bound_states::BoundProblemBuilder, potentials::potential::Potential, quantum::{
            problem_selector::{get_args, ProblemSelector}, problems_impl, units::{Au, Energy}, utility::linspace
        }, utility::save_data
    },
};
use shielded_molecules::{SystemParams, get_problem};

fn main() {
    Problems::select(&mut get_args());
}

pub struct Problems;

problems_impl!(Problems, "shielded molecules",
    "potential" => |_| Self::potential(),
    "bound states" => |_| Self::bound_states(),
);

impl Problems {
    pub fn potential() {
        let params = default_params();
        let problem = get_problem(&params);

        let distances = linspace(2., 100., 101);

        let values = distances
            .iter()
            .map(|&r| {
                let mut pot_mat = Mat::zeros(problem.potential.size(), problem.potential.size());
                problem.potential.value_inplace(r, &mut pot_mat);

                pot_mat[(0, 0)]
            })
            .collect();

        save_data("potential_value", "r\tpot_value", &[distances, values]).unwrap();
    }

    pub fn bound_states() {
        let params = default_params();
        let problem = get_problem(&params);
        let step_rule = LocalWavelengthStepRule::new(1e-4, 10., 500.);
        let r_range = (2., 10., 500.);
        let e_range = (0., 1.);
        let e_err = 1e-3;

        let bound_states = BoundProblemBuilder::new(&problem.particles, &problem.potential)
                .with_propagation(step_rule.clone(), Johnson)
                .with_range(r_range.0, r_range.1, r_range.2)
                .build();

        let e_range = (Energy(e_range.0, Au), Energy(e_range.1, Au));

        let bound_states = bound_states.bound_states(e_range, Energy(e_err, Au));

        println!("{bound_states:?}");
    }
}

pub fn default_params() -> SystemParams {
    SystemParams {
        l_max: 0,
        parity: ParityBlock::Positive,

        c6: 1e3,
        c3: 1e3,
        ksi: 1.,

        trap_freq: 1e-3,

        mass: 1.,
        energy: 1.,
    }
}
