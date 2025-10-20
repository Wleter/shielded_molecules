pub use cc_problems;
use cc_problems::prelude::*;
use cc_problems::{
    coupled_chan::{
        coupling::{composite::Composite, masked::Masked, Asymptote, Levels, RedCoupling}, dispersion::Dispersion, Operator
    }, hilbert_space::{
        cast_variant, dyn_space::{SpaceBasis, SubspaceBasis}, filter_space, operator::kron_delta, operator_mel, Parity
    }, operator_mel::p1_factor, spin_algebra::{clebsch_gordan, get_spin_basis}
};

#[derive(Clone, Copy, Debug)]
pub struct SystemParams {
    pub l_max: u32,
    pub parity: Parity,
    pub symmetry: Parity,
    pub entrance: (u32, i32),

    pub c6: f64,
    pub c3: f64,
    pub ksi: f64,

    pub trap_freq: f64,

    pub mass: f64,
    pub energy: f64,
}

pub struct SystemProblem {
    pub red_coupling: RedCoupling<Composite<Masked<Dispersion>>>,
    pub channels: Vec<(u32, i32)>,
}

pub fn get_problem(params: &SystemParams) -> SystemProblem {
    let angular = (0..=params.l_max)
        .flat_map(|n_tot| get_spin_basis(n_tot.into()))
        .collect();

    let mut states = SpaceBasis::default();
    let l = states.push_subspace(SubspaceBasis::new(angular));

    let basis = filter_space!(dyn states, |[l: Spin]| {
        let m = l.m;
        let l = l.s;

        let m_parity = match params.parity {
            Parity::Even => m.double_value().abs() % 4 == 0,
            Parity::Odd => m.double_value().abs() % 4 == 2,
            Parity::All => true,
        };
        let l_parity = match params.symmetry {
            Parity::All => true,
            Parity::Odd => l.double_value() % 4 == 2,
            Parity::Even => l.double_value() % 4 == 0,
        };

        m_parity && l_parity
    });

    let c3_dispersion = Dispersion::new(2. * params.c3, -3);
    let c6_dispersion = Dispersion::new(2. * params.c6, -6);
    let trap = Dispersion::new(0.5 * params.mass * params.trap_freq, 2);

    let sin_2ksi = (2. * params.ksi).sin();

    let c3_mask = operator_mel!(dyn basis, [l], |[l: Spin]| {
        let factor = p1_factor(l.ket.s) / p1_factor(l.bra.s)
            * clebsch_gordan(l.ket.s, hi32!(0), hu32!(2), hi32!(0), l.bra.s, hi32!(0));

        let first_term = clebsch_gordan(l.ket.s, l.ket.m, hu32!(2), hi32!(0), l.bra.s, l.bra.m);
        let second_term = 1.5f64.sqrt() * sin_2ksi
            * (clebsch_gordan(l.ket.s, l.ket.m, hu32!(2), hi32!(2), l.bra.s, l.bra.m) +
                clebsch_gordan(l.ket.s, l.ket.m, hu32!(2), hi32!(-2), l.bra.s, l.bra.m));

        factor * (first_term + second_term)
    });

    let c6_mask = operator_mel!(dyn basis, [l], |[l: Spin]| {
        let factor = p1_factor(l.ket.s) / p1_factor(l.bra.s);

        let first_term = 0.4 * (1. - sin_2ksi.powi(2) / 3.) * kron_delta([l]);
        let second_term = -clebsch_gordan(l.ket.s, hi32!(0), hu32!(2), hi32!(0), l.bra.s, hi32!(0)) / 7.;
        let first_sub_term = 2. * (1. - 2. / 3. * sin_2ksi.powi(2))
            * clebsch_gordan(l.ket.s, l.ket.m, hu32!(2), hi32!(0), l.bra.s, l.bra.m);
        let second_sub_term = (2f64 / 3.).sqrt() * sin_2ksi
            * (clebsch_gordan(l.ket.s, l.ket.m, hu32!(2), hi32!(2), l.bra.s, l.bra.m)
                + clebsch_gordan(l.ket.s, l.ket.m, hu32!(2), hi32!(-2), l.bra.s, l.bra.m));
        let second_term = second_term * (first_sub_term + second_sub_term);

        let third_term = -(2f64 / 35.).sqrt() * clebsch_gordan(l.ket.s, hi32!(0), hu32!(4), hi32!(0), l.bra.s, hi32!(0));
        let first_sub_term = (2f64 / 35.).sqrt() * (2. + sin_2ksi.powi(2))
            * clebsch_gordan(l.ket.s, l.ket.m, hu32!(4), hi32!(0), l.bra.s, l.bra.m);
        let second_sub_term = (2. / 7f64.sqrt()) * sin_2ksi
            * (clebsch_gordan(l.ket.s, l.ket.m, hu32!(4), hi32!(2), l.bra.s, l.bra.m)
                + clebsch_gordan(l.ket.s, l.ket.m, hu32!(4), hi32!(-2), l.bra.s, l.bra.m));
        let third_sub_term = sin_2ksi.powi(2)
            * (clebsch_gordan(l.ket.s, l.ket.m, hu32!(4), hi32!(4), l.bra.s, l.bra.m)
                + clebsch_gordan(l.ket.s, l.ket.m, hu32!(4), hi32!(-4), l.bra.s, l.bra.m));
        let third_term = third_term * (first_sub_term + second_sub_term + third_sub_term);

        factor * (first_term + second_term + third_term)
    });

    let trap_mask = Operator::identity(basis.len());

    let potential = Composite::new(vec![
        Masked::new(trap, trap_mask),
        Masked::new(c3_dispersion, c3_mask),
        Masked::new(c6_dispersion, c6_mask),
    ]);

    let channels: Vec<(u32, i32)> = basis.elements_indices
        .iter()
        .map(|x| {
            let lm = cast_variant!(dyn x.index(l, &basis.basis), Spin);

            (lm.s.double_value() / 2, lm.m.double_value() / 2)
        })
        .collect();

    let l = basis.elements_indices
        .iter()
        .map(|x| cast_variant!(dyn x.index(l, &basis.basis), Spin).s.double_value() / 2)
        .collect();

    let levels = Levels {
        l,
        asymptote: vec![0.; basis.len()],
    };

    let asymptote = Asymptote::new_diagonal(
        params.mass * AuMass,
        params.energy * AuEnergy,
        levels,
        channels.iter().enumerate().find(|x| *x.1 == params.entrance).unwrap().0
    );

    let red_coupling = RedCoupling::new(potential, asymptote);

    SystemProblem {
        red_coupling,
        channels,
    }
}
