use cc_problems::{
    hilbert_space::{Parity, operator::Operator},
    prelude::*,
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
    pub fn potential() -> Result<()> {
        let params = default_params();
        let problem = get_problem(&params);

        let distances = linspace(0.5, 6., 1001);

        let saver = DataSaver::new(
            "data/potential_value.dat",
            DatFormat::new("r\tpot_value"),
            FileAccess::Create,
        )?;

        distances.iter().for_each(|&r| {
            let mut pot_mat = Operator::zeros(problem.red_coupling.size());
            problem.red_coupling.value_inplace(r, &mut pot_mat);

            let value = -pot_mat[(0, 0)] / 2.;

            saver.send([r, value]);
        });

        Ok(())
    }

    pub fn bound_states() -> Result<()> {
        let params = default_params();
        let problem = get_problem(&params);
        let step_rule = LocalWavelengthStep::new(1e-4, 10., 500.);
        let r_range = [0.6 * Bohr, 1.5 * Bohr, 6. * Bohr];
        let e_range = [0., 5.];
        let e_err = 1e-4;

        let bound_finder = BoundStatesFinder::default()
            .set_propagator(|b, w| JohnsonLogDerivative::new(w, step_rule.into(), b))
            .set_r_range(r_range)
            .set_parameter_range(e_range, e_err)
            .set_node_range(NodeRangeTarget::Range(0, 5))
            .set_problem(|e| {
                let mut problem = problem.red_coupling.clone();
                problem.asymptote.set_energy(e * AuEnergy);

                problem
            });

        let bound_states = bound_finder.bound_states();
        for b in bound_states {
            println!("{b:?}");
        }

        Ok(())
    }
}

pub fn default_params() -> SystemParams {
    SystemParams {
        l_max: 2,
        parity: Parity::All,
        symmetry: Parity::All,
        entrance: (0, 0),

        c3: 9.25010281855974,
        c6: 13.543607688069887,
        ksi: 0.,

        trap_freq: 1.,

        mass: 1.,
        energy: 0.,
    }
}
