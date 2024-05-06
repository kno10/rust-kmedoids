use crate::arrayadapter::ArrayAdapter;
use core::ops::AddAssign;
use num_traits::{Signed, Zero};

/// Object id and distance pair
#[derive(Debug, Copy, Clone)]
pub(crate) struct DistancePair<N> {
	pub(crate) i: u32,
	pub(crate) d: N,
}
impl<N> DistancePair<N> {
	pub(crate) fn new(i: u32, d: N) -> Self {
		DistancePair { i, d }
	}
}
impl<N: Zero> DistancePair<N> {
	pub(crate) fn empty() -> Self {
		DistancePair {
			i: u32::MAX,
			d: N::zero(),
		}
	}
}

/// Information kept for each point: two such pairs
#[derive(Debug, Copy, Clone)]
pub(crate) struct Rec<N> {
	pub(crate) near: DistancePair<N>,
	pub(crate) seco: DistancePair<N>,
}
impl<N> Rec<N> {
	pub(crate) fn new(i1: u32, d1: N, i2: u32, d2: N) -> Rec<N> {
		Rec {
			near: DistancePair { i: i1, d: d1 },
			seco: DistancePair { i: i2, d: d2 },
		}
	}
}
impl<N: Zero> Rec<N> {
	pub(crate) fn empty() -> Self {
		Rec {
			near: DistancePair::empty(),
			seco: DistancePair::empty(),
		}
	}
}

/// Information kept for each point: three such pairs
#[derive(Debug, Copy, Clone)]
pub(crate) struct Reco<N> {
	pub(crate) near: DistancePair<N>,
	pub(crate) seco: DistancePair<N>,
	pub(crate) third: DistancePair<N>,
}
impl<N> Reco<N> {
	pub(crate) fn new(i1: u32, d1: N, i2: u32, d2: N, i3: u32, d3: N) -> Reco<N> {
		Reco {
			near: DistancePair { i: i1, d: d1 },
			seco: DistancePair { i: i2, d: d2 },
			third: DistancePair { i: i3, d: d3 },
		}
	}
}
impl<N: Zero> Reco<N> {
	pub(crate) fn empty() -> Self {
		Reco {
			near: DistancePair::empty(),
			seco: DistancePair::empty(),
			third: DistancePair::empty(),
		}
	}
}

/// Find the minimum (index and value)
#[inline]
pub(crate) fn find_min<'a, L: 'a, I: 'a>(a: &mut I) -> (usize, L)
	where
		L: PartialOrd + Clone + Zero,
		I: std::iter::Iterator<Item = &'a L>,
{
	let mut a = a.enumerate();
	let mut best: (usize, L) = (0, a.next().unwrap().1.clone());
	for (ik, iv) in a {
		if *iv < best.1 {
			best = (ik, iv.clone());
		}
	}
	best
}

/// Find the maximum (index and value)
#[inline]
pub(crate) fn find_max<'a, L: 'a, I: 'a>(a: &mut I) -> (usize, L)
	where
		L: PartialOrd + Clone + Zero,
		I: std::iter::Iterator<Item = &'a L>,
{
	let mut a = a.enumerate();
	let mut best: (usize, L) = (0, a.next().unwrap().1.clone());
	for (ik, iv) in a {
		if *iv > best.1 {
			best = (ik, iv.clone());
		}
	}
	best
}

/// Choose the best medoid within a partition
/// Used by ther alternating algorithm, or when a single cluster is requested.
pub(crate) fn choose_medoid_within_partition<M, N, L>(
	mat: &M,
	assi: &[usize],
	med: &mut [usize],
	m: usize,
) -> (bool, L)
	where
		N: PartialOrd + Clone,
		L: AddAssign + Signed + Zero + PartialOrd + Clone + From<N>,
		M: ArrayAdapter<N>,
{
	let first = med[m];
	let mut best = first;
	let mut sumb = L::zero();
	for (i, &a) in assi.iter().enumerate() {
		if first != i && a == m {
			sumb += L::from(mat.get(first, i));
		}
	}
	for (j, &aj) in assi.iter().enumerate() {
		if j != first && aj == m {
			let mut sumj = L::zero();
			for (i, &ai) in assi.iter().enumerate() {
				if i != j && ai == m {
					sumj += L::from(mat.get(j, i));
				}
			}
			if sumj < sumb {
				best = j;
				sumb = sumj;
			}
		}
	}
	med[m] = best;
	(best != first, sumb)
}

/// Debug helper function
pub(crate) fn debug_assert_assignment<M, N>(_mat: &M, _med: &[usize], _data: &[Rec<N>])
	where
		N: PartialOrd + Clone,
		M: ArrayAdapter<N>,
{
	#[cfg(feature = "assertions")]
	for o in 0.._mat.len() {
		debug_assert!(
			_mat.get(o, _med[_data[o].near.i as usize]) == _data[o].near.d,
			"primary assignment inconsistent"
		);
		debug_assert!(
			_mat.get(o, _med[_data[o].seco.i as usize]) == _data[o].seco.d,
			"secondary assignment inconsistent"
		);
		debug_assert!(
			_data[o].near.d <= _data[o].seco.d,
			"nearest is farther than second nearest"
		);
	}
}

/// Debug helper function, for methods with three nearest medoids
pub(crate) fn debug_assert_assignment_th<M, N>(_mat: &M, _med: &[usize], _data: &[Reco<N>])
	where
		N: PartialOrd + Clone,
		M: ArrayAdapter<N>,
{
	#[cfg(feature = "assertions")]
	for o in 0.._mat.len() {
		debug_assert!(
			_mat.get(o, _med[_data[o].near.i as usize]) == _data[o].near.d,
			"primary assignment inconsistent"
		);
		debug_assert!(
			_mat.get(o, _med[_data[o].seco.i as usize]) == _data[o].seco.d,
			"secondary assignment inconsistent"
		);
		debug_assert!(
			_mat.get(o, _med[_data[o].third.i as usize]) == _data[o].third.d,
			"third assignment inconsistent"
		);
		debug_assert!(
			_data[o].near.d <= _data[o].seco.d,
			"nearest is farther than second nearest"
		);
		debug_assert!(
			_data[o].seco.d <= _data[o].third.d,
			"second nearest is farther than third nearest"
		);
	}
}

/// test two arrays for equality, used in tests only
#[cfg(test)]
pub(crate) fn assert_array(result: Vec<usize>, expect: Vec<usize>, msg: &'static str) {
	assert!(
		result.iter().zip(expect.iter()).all(|(a, b)| a == b),
		"{}",
		msg
	);
}
