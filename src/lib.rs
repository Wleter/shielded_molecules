use faer::{Mat, traits::num_traits::Float};
use scattering_problems::{
    alkali_rotor_atom::ParityBlock,
    scattering_solver::{
        boundary::Asymptotic,
        potentials::{
            composite_potential::Composite, dispersion_potential::Dispersion,
            masked_potential::MaskedPotential,
        },
        quantum::{
            cast_variant,
            clebsch_gordan::{clebsch_gordan, hi32, hu32},
            operator_mel,
            params::{particle::Particle, particles::Particles},
            states::{
                States, StatesBasis,
                braket::kron_delta,
                operator::Operator,
                spins::{Spin, get_spin_basis},
                state::{StateBasis, into_variant},
            },
            units::{Au, Energy, Mass},
        },
        utility::AngMomentum,
    },
    utility::p1_factor,
};

pub use scattering_problems;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Basis {
    Angular(Spin),
}

#[derive(Clone, Copy, Debug)]
pub struct SystemParams {
    pub l_max: u32,
    pub parity: ParityBlock,

    pub c6: f64,
    pub c3: f64,
    pub ksi: f64,

    pub trap_freq: f64,

    pub mass: f64,
    pub energy: f64,
}

pub struct SystemProblem {
    pub particles: Particles,
    pub potential: Composite<MaskedPotential<Mat<f64>, Dispersion>>,
    pub basis: StatesBasis<Basis>,
}

pub fn get_problem(params: &SystemParams) -> SystemProblem {
    let angular = (0..=params.l_max)
        .flat_map(|n_tot| into_variant(get_spin_basis(n_tot.into()), Basis::Angular))
        .collect();

    let mut states = States::default();
    states.push_state(StateBasis::new(angular));

    let basis = states
        .iter_elements()
        .filter(|x| {
            let m = cast_variant!(x[0], Basis::Angular).ms;

            match params.parity {
                ParityBlock::Positive => m.double_value().abs() % 4 == 0,
                ParityBlock::Negative => m.double_value().abs() % 4 == 2,
                ParityBlock::All => true,
            }
        })
        .collect();

    let c3_dispersion = Dispersion::new(2. * params.c3, -3);
    let c6_dispersion = Dispersion::new(2. * params.c6, -6);
    let trap = Dispersion::new(0.5 * params.mass * params.trap_freq, 2);

    let sin_2ksi = (2. * params.ksi).sin();

    let c3_mask = operator_mel!(&basis, |[l: Basis::Angular]| {
        let factor = p1_factor(l.ket.s) / p1_factor(l.bra.s)
            * clebsch_gordan(l.ket.s, hi32!(0), hu32!(2), hi32!(0), l.bra.s, hi32!(0));

        let first_term = clebsch_gordan(l.ket.s, l.ket.ms, hu32!(2), hi32!(0), l.bra.s, l.bra.ms);
        let second_term = 1.5.sqrt() * sin_2ksi
            * (clebsch_gordan(l.ket.s, l.ket.ms, hu32!(2), hi32!(2), l.bra.s, l.bra.ms) +
                clebsch_gordan(l.ket.s, l.ket.ms, hu32!(2), hi32!(-2), l.bra.s, l.bra.ms));

        factor * (first_term + second_term)
    });

    let c6_mask = operator_mel!(&basis, |[l: Basis::Angular]| {
        let factor = p1_factor(l.ket.s) / p1_factor(l.bra.s);

        let first_term = 0.4 * (1. - sin_2ksi.powi(2) / 3.) * kron_delta([l]);
        let second_term = -clebsch_gordan(l.ket.s, hi32!(0), hu32!(2), hi32!(0), l.bra.s, hi32!(0)) / 7.;
        let first_sub_term = 2. * (1. - 2. / 3. * sin_2ksi.powi(2))
            * clebsch_gordan(l.ket.s, l.ket.ms, hu32!(2), hi32!(0), l.bra.s, l.bra.ms);
        let second_sub_term = (2. / 3.).sqrt() * sin_2ksi
            * (clebsch_gordan(l.ket.s, l.ket.ms, hu32!(2), hi32!(2), l.bra.s, l.bra.ms)
                + clebsch_gordan(l.ket.s, l.ket.ms, hu32!(2), hi32!(-2), l.bra.s, l.bra.ms));
        let second_term = second_term * (first_sub_term + second_sub_term);

        let third_term = -(2. / 35.).sqrt() * clebsch_gordan(l.ket.s, hi32!(0), hu32!(4), hi32!(0), l.bra.s, hi32!(0));
        let first_sub_term = (2. / 35.).sqrt() * (2. + sin_2ksi.powi(2))
            * clebsch_gordan(l.ket.s, l.ket.ms, hu32!(4), hi32!(0), l.bra.s, l.bra.ms);
        let second_sub_term = (2. / 7.0.sqrt()) * sin_2ksi
            * (clebsch_gordan(l.ket.s, l.ket.ms, hu32!(4), hi32!(2), l.bra.s, l.bra.ms)
                + clebsch_gordan(l.ket.s, l.ket.ms, hu32!(4), hi32!(-2), l.bra.s, l.bra.ms));
        let third_sub_term = sin_2ksi.powi(2)
            * (clebsch_gordan(l.ket.s, l.ket.ms, hu32!(4), hi32!(4), l.bra.s, l.bra.ms)
                + clebsch_gordan(l.ket.s, l.ket.ms, hu32!(4), hi32!(-4), l.bra.s, l.bra.ms));
        let third_term = third_term * (first_sub_term + second_sub_term + third_sub_term);

        factor * (first_term + second_term + third_term)
    });

    let trap_mask = Operator::new(Mat::identity(basis.len(), basis.len()));

    let potential = Composite::from_vec(vec![
        MaskedPotential::new(trap, trap_mask.into_backed()),
        MaskedPotential::new(c3_dispersion, c3_mask.into_backed()),
        MaskedPotential::new(c6_dispersion, c6_mask.into_backed()),
    ]);

    let centrifugal = basis
        .iter()
        .map(|a| {
            let l = cast_variant!(a[0], Basis::Angular).s;
            assert!(
                l.double_value() & 1 == 0,
                "Angular momentum is not an integer"
            );

            AngMomentum(l.double_value() / 2)
        })
        .collect();
    let energies = vec![0.; basis.len()];

    let asymptotic = Asymptotic {
        centrifugal,
        entrance: 0,
        channel_energies: energies,
        channel_states: Mat::identity(basis.len(), basis.len()),
    };

    let particle = Particle::new("molecule", Mass(2. * params.mass, Au));
    let mut particles = Particles::new_pair(particle.clone(), particle, Energy(params.energy, Au));
    particles.insert(asymptotic);

    SystemProblem {
        particles,
        potential,
        basis,
    }
}
