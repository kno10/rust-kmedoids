//! k-Medoids Clustering with the FasterPAM Algorithm
//!
//! Given a dissimilarity matrix of size n x n, use:
//! ```
//! let mut rng = rand::thread_rng();
//! let mut meds = random_initialization(n, k, &mut rng);
//! let (loss, numswap, numiter, assignment) = fasterpam(data, &mut meds, 100);
//! ```
pub mod safeadd;

pub use crate::safeadd::SafeAdd;

use ndarray::Array2;
use num_traits::{NumAssignOps, Signed, Zero};

/// Object id and distance pair
#[derive(Debug, Copy, Clone)]
struct DistancePair<N> {
	i: u32,
	d: N,
}
// Information kept for each point: two such pairs
#[derive(Debug)]
struct PointInformation<N> {
	near: DistancePair<N>,
	seco: DistancePair<N>,
}

/// Perform the initial assignment to medoids
fn initial_assignment<N: NumAssignOps + Zero + PartialOrd + Copy + SafeAdd>(
	mat: &Array2<N>,
	med: &Vec<usize>,
	data: &mut Vec<PointInformation<N>>,
) -> N {
	let n = mat.shape()[0];
	let k = med.len();
	let firstcenter = med[0];
	let unassigned = k as u32;
	let mut loss: N = N::zero();
	for i in 0..n {
		let mut cur = PointInformation::<N> {
			near: DistancePair {
				i: 0,
				d: mat[[i, firstcenter]],
			},
			seco: DistancePair {
				i: unassigned,
				d: N::zero(),
			},
		};
		for m in 1..k {
			let d = mat[[i, med[m]]];
			if d < cur.near.d || i == med[m] {
				cur.seco = cur.near;
				cur.near = DistancePair { i: m as u32, d: d };
			} else if cur.seco.i == unassigned || d < cur.seco.d {
				cur.seco = DistancePair { i: m as u32, d: d };
			}
		}
		loss.safe_inc(cur.near.d);
		data.push(cur);
	}
	return loss;
}

/// Update the loss when removing each medoid
fn update_removal_loss<N: NumAssignOps + Signed + Copy + Zero + SafeAdd>(
	data: &Vec<PointInformation<N>>,
	loss: &mut Vec<N>,
) {
	let n = data.len();
	// not yet stable API: loss.fill(N::zero());
	for i in 0..loss.len() {
		loss[i] = N::zero();
	}
	for i in 0..n {
		let rec = &data[i];
		loss[rec.near.i as usize].safe_inc(rec.seco.d - rec.near.d);
	}
}

/// Find the minimum (both index and value)
#[inline]
fn find_min<N: PartialOrd + Copy + Zero>(a: &Vec<N>) -> (usize, N) {
	let mut rk: usize = a.len();
	let mut rv: N = N::zero();
	for (ik, iv) in a.iter().enumerate() {
		if ik == 0 || *iv < rv {
			rk = ik;
			rv = *iv;
		}
	}
	return (rk, rv);
}

/// Update the second nearest medoid information
///
/// Called after each swap.
fn update_second_nearest<N: NumAssignOps + PartialOrd + Copy + SafeAdd>(
	mat: &Array2<N>,
	med: &Vec<usize>,
	n: usize,
	b: usize,
	o: usize,
	djo: N,
) -> DistancePair<N> {
	let mut s = DistancePair {
		i: b as u32,
		d: djo,
	};
	for i in 0..med.len() {
		if i == n || i == b {
			continue;
		}
		let dm = mat[[o, med[i]]];
		if dm < s.d {
			s = DistancePair { i: i as u32, d: dm };
		}
	}
	return s;
}
/// Find the best swap
fn find_best_swap<N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd>(
	mat: &Array2<N>,
	removal_loss: &Vec<N>,
	data: &Vec<PointInformation<N>>,
	j: usize,
) -> (usize, N) {
	let n = mat.shape()[0];
	let mut ploss = removal_loss.clone();
	let mut acc = N::zero();
	for o in 0..n {
		let reco = &data[o];
		let djo = mat[[j, o]];
		// New medoid is closest:
		if djo < reco.near.d {
			acc.safe_inc(djo - reco.near.d);
			// loss already includes ds - dn, remove
			ploss[reco.near.i as usize].safe_inc(reco.near.d - reco.seco.d);
		} else if djo < reco.seco.d {
			// loss already includes ds - dn, adjust to d(xo) - dn
			ploss[reco.near.i as usize].safe_inc(djo - reco.seco.d);
		}
	}
	let (b, bloss) = find_min(&ploss);
	return (b, bloss + acc);
}

/// Debug helper function
#[cfg(feature = "assertions")]
fn debug_validate_assignment<N: NumAssignOps + PartialOrd + Copy + SafeAdd>(
	mat: &Array2<N>,
	med: &Vec<usize>,
	data: &Vec<PointInformation<N>>,
) {
	let n = mat.shape()[0];
	for o in 0..n {
		debug_assert!(
			mat[[o, med[data[o].near.i as usize]]] == data[o].near.d,
			"primary assignment inconsistent"
		);
		debug_assert!(
			mat[[o, med[data[o].seco.i as usize]]] == data[o].seco.d,
			"secondary assignment inconsistent"
		);
		debug_assert!(data[o].near.d <= data[o].seco.d);
	}
}

/// Perform a single swap
fn do_swap<N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd>(
	mat: &Array2<N>,
	med: &mut Vec<usize>,
	data: &mut Vec<PointInformation<N>>,
	b: usize,
	j: usize,
) -> N {
	let n = mat.shape()[0];
	med[b] = j;
	let mut newloss = N::zero();
	for o in 0..n {
		let mut reco = &mut data[o];
		if o == j {
			if reco.near.i != b as u32 {
				reco.seco = reco.near;
			}
			reco.near = DistancePair {
				i: b as u32,
				d: N::zero(),
			};
			continue;
		}
		let djo = mat[[j, o]];
		// Nearest medoid is gone:
		if reco.near.i == b as u32 {
			if djo < reco.seco.d {
				reco.near = DistancePair {
					i: b as u32,
					d: djo,
				};
			} else {
				reco.near = reco.seco;
				reco.seco = update_second_nearest(&mat, &med, reco.near.i as usize, b, o, djo);
			}
		} else {
			// nearest not removed
			if djo < reco.near.d {
				reco.seco = reco.near;
				reco.near = DistancePair {
					i: b as u32,
					d: djo,
				};
			} else if reco.seco.i == b as u32 {
				// second nearest was replaced
				reco.seco = update_second_nearest(&mat, &med, reco.near.i as usize, b, o, djo);
			} else if djo < reco.seco.d {
				reco.seco = DistancePair {
					i: b as u32,
					d: djo,
				};
			}
		}
		newloss.safe_inc(reco.near.d);
	}
	#[cfg(feature = "assertions")]
	debug_validate_assignment(&mat, &med, &data);
	return newloss;
}

/// Random initialization
///
/// Given a dissimilarity matrix of size n x n, use:
/// ```
/// let mut rng = rand::thread_rng();
/// let mut meds = random_initialization(n, k, &mut rng);
/// let (loss, numswap, numiter, assignment) = fasterpam(data, &mut meds, 100);
/// ```
///
/// * `n` - size of the data set
/// * `k` - number of clusters to find
/// * `rng` - random number generator
pub fn random_initialization(n: usize, k: usize, mut rng: &mut impl rand::Rng) -> Vec<usize> {
	return rand::seq::index::sample(&mut rng, n, k).into_vec();
}

/// Run the FasterPAM algorithm.
///
/// Given a dissimilarity matrix of size n x n, use:
/// ```
/// let mut rng = rand::thread_rng();
/// let mut meds = random_initialization(n, k, &mut rng);
/// let (loss, numswap, numiter, assignment) = fasterpam(data, &mut meds, 100);
/// ```
///
/// * type `N` - some signed data type such as `i32` or `f64`
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
///
/// returns a tuple containing:
/// * the final loss
/// * the number of swaps performed
/// * the number of iterations needed
/// * the final cluster assignment
pub fn fasterpam<N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd>(
	mat: &Array2<N>,
	mut med: &mut Vec<usize>,
	maxiter: usize,
) -> (N, usize, usize, Vec<usize>) {
	let n = mat.shape()[0];
	let k = med.len();
	assert_eq!(mat.shape()[1], n);
	assert!(k <= n);
	let mut data = Vec::<PointInformation<N>>::with_capacity(n);
	let mut loss = initial_assignment(&mat, &med, &mut data);
	#[cfg(feature = "assertions")]
	debug_validate_assignment(&mat, &med, &data);
	// println!("Initial loss is {}", loss);
	let mut removal_loss = vec![N::zero(); k];
	let mut lastswap = n;
	let mut numswaps = 0;
	let mut iter = 0;
	while iter < maxiter {
		iter += 1;
		// println!("Iteration {} before {}", iter, loss);
		let swaps_before = numswaps;
		update_removal_loss(&data, &mut removal_loss);
		for j in 0..n {
			if j == lastswap {
				break;
			}
			if j == data[j].near.i as usize {
				continue; // This already is a medoid
			}
			let (b, change) = find_best_swap(&mat, &removal_loss, &data, j);
			if change >= N::zero() {
				continue; // No improvement
			}
			numswaps += 1;
			lastswap = j;
			// perform the swap
			let newloss = do_swap(&mat, &mut med, &mut data, b, j);
			// println!("{} + {} = {} vs. {}", loss, change, loss + change, newloss);
			if newloss >= loss {
				break; // Probably numerically unstable now.
			}
			loss = newloss;
			update_removal_loss(&data, &mut removal_loss);
		}
		if numswaps == swaps_before {
			break; // converged
		}
	}
	// println!("final loss: {}", loss);
	// println!("number of swaps: {}", numswaps);
	let assignment = data.iter().map(|x| x.near.i as usize).collect();
	return (loss, numswaps, iter, assignment);
}
