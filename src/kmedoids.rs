//! k-Medoids Clustering with the FasterPAM Algorithm
//!
//! For details on the implemented FasterPAM algorithm, please see:
//!
//! Erich Schubert, Peter J. Rousseeuw  
//! **Fast and Eager k-Medoids Clustering:  
//! O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms**  
//! Information Systems (101), 2021, 101804  
//! <https://doi.org/10.1016/j.is.2021.101804> (open access)
//!
//! Erich Schubert, Peter J. Rousseeuw:  
//! **Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms**  
//! In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.  
//! <https://doi.org/10.1007/978-3-030-32047-8_16>  
//! Preprint: <https://arxiv.org/abs/1810.05691>
//!
//! This is a port of the original Java code from [ELKI](https://elki-project.github.io/) to Rust.
//! But it does not include all functionality in the original benchmarks.
//!
//! If you use this in scientific work, please consider citing above articles.
//!
//! ## Example
//!
//! Given a dissimilarity matrix of size 4 x 4, use:
//! ```
//! let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
//! let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
//! let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::fasterpam(&data, &mut meds, 100);
//! println!("Loss is: {}", loss);
//! ```
pub mod arrayadapter;

pub use crate::arrayadapter::ArrayAdapter;
use core::ops::{AddAssign, Div, Sub};
use num_traits::{Signed, Zero};
use std::convert::From;

/// Object id and distance pair
#[derive(Debug, Copy, Clone)]
struct DistancePair<N> {
	i: u32,
	d: N,
}
// Information kept for each point: two such pairs
#[derive(Debug)]
struct Rec<N> {
	near: DistancePair<N>,
	seco: DistancePair<N>,
}

/// Perform the initial assignment to medoids
#[inline]
fn initial_assignment<M, N, L>(mat: &M, med: &[usize], data: &mut Vec<Rec<N>>) -> L
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	assert!(data.is_empty(), "data not empty");
	let firstcenter = med[0];
	let mut loss: L = L::zero();
	for i in 0..n {
		let mut cur = Rec::<N> {
			near: DistancePair {
				i: 0,
				d: mat.get(i, firstcenter),
			},
			seco: DistancePair {
				i: u32::MAX,
				d: N::zero(),
			},
		};
		for (m, &me) in med.iter().enumerate().skip(1) {
			let dm = mat.get(i, me);
			if dm < cur.near.d || i == me {
				cur.seco = cur.near;
				cur.near = DistancePair { i: m as u32, d: dm };
			} else if cur.seco.i == u32::MAX || dm < cur.seco.d {
				cur.seco = DistancePair { i: m as u32, d: dm };
			}
		}
		loss += L::from(cur.near.d);
		data.push(cur);
	}
	loss
}

/// Update the loss when removing each medoid
fn update_removal_loss<N, L>(data: &[Rec<N>], loss: &mut Vec<L>)
where
	N: Zero + Copy,
	L: AddAssign + Signed + Copy + Zero + From<N>,
{
	loss.fill(L::zero()); // stable since 1.50
	for rec in data.iter() {
		loss[rec.near.i as usize] += L::from(rec.seco.d) - L::from(rec.near.d);
		// as N might be unsigned
	}
}

/// Find the minimum (both index and value)
#[inline]
fn find_min<L>(a: &[L]) -> (usize, L)
where
	L: PartialOrd + Copy + Zero,
{
	let mut best: (usize, L) = (a.len(), L::zero());
	for (ik, &iv) in a.iter().enumerate() {
		if ik == 0 || iv < best.1 {
			best = (ik, iv);
		}
	}
	best
}

/// Update the second nearest medoid information
///
/// Called after each swap.
#[inline]
fn update_second_nearest<M, N>(
	mat: &M,
	med: &[usize],
	n: usize,
	b: usize,
	o: usize,
	djo: N,
) -> DistancePair<N>
where
	N: PartialOrd + Copy,
	M: ArrayAdapter<N>,
{
	let mut s = DistancePair {
		i: b as u32,
		d: djo,
	};
	for (i, &mi) in med.iter().enumerate() {
		if i == n || i == b {
			continue;
		}
		let dm = mat.get(o, mi);
		if dm < s.d {
			s = DistancePair { i: i as u32, d: dm };
		}
	}
	s
}
/// Find the best swap for object j - FastPAM version
#[inline]
fn find_best_swap<M, N, L>(mat: &M, removal_loss: &[L], data: &[Rec<N>], j: usize) -> (L, usize)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let mut ploss = removal_loss.to_vec();
	// Improvement from the journal version:
	let mut acc = L::zero();
	for (o, reco) in data.iter().enumerate() {
		let djo = mat.get(j, o);
		// New medoid is closest:
		if djo < reco.near.d {
			acc += L::from(djo) - L::from(reco.near.d);
			// loss already includes ds - dn, remove
			ploss[reco.near.i as usize] += L::from(reco.near.d) - L::from(reco.seco.d);
		} else if djo < reco.seco.d {
			// loss already includes ds - dn, adjust to d(xo) - dn
			ploss[reco.near.i as usize] += L::from(djo) - L::from(reco.seco.d);
		}
	}
	let (b, bloss) = find_min(&ploss);
	(bloss + acc, b) // add the shared accumulator
}

/// Find the best swap for object j - slower PAM version
#[inline]
fn find_best_swap_pam<M, N, L>(mat: &M, med: &[usize], data: &[Rec<N>], j: usize) -> (L, usize)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let recj = &data[j];
	let mut best = (L::zero(), usize::MAX);
	for (m, _) in med.iter().enumerate() {
		let mut acc: L = -L::from(recj.near.d); // j becomes medoid
		for (o, reco) in data.iter().enumerate() {
			if o == j {
				continue;
			}
			let djo = mat.get(j, o);
			// Current medoid is being replaced:
			if reco.near.i as usize == m {
				if djo < reco.seco.d {
					// Assign to new medoid:
					acc += L::from(djo) - L::from(reco.near.d)
				} else {
					// Assign to second nearest instead:
					acc += L::from(reco.seco.d) - L::from(reco.near.d)
				}
			} else if djo < reco.near.d {
				// new mediod is closer:
				acc += L::from(djo) - L::from(reco.near.d)
			} // else no change
		}
		if acc < best.0 {
			best = (acc, m);
		}
	}
	best
}

/// Debug helper function
#[cfg(feature = "assertions")]
fn debug_validate_assignment<M, N>(mat: &M, med: &[usize], data: &[Rec<N>])
where
	N: PartialOrd + Copy,
	M: ArrayAdapter<N>,
{
	for o in 0..mat.len() {
		debug_assert!(
			mat.get(o, med[data[o].near.i as usize]) == data[o].near.d,
			"primary assignment inconsistent"
		);
		debug_assert!(
			mat.get(o, med[data[o].seco.i as usize]) == data[o].seco.d,
			"secondary assignment inconsistent"
		);
		debug_assert!(
			data[o].near.d <= data[o].seco.d,
			"nearest is farther than second nearest"
		);
	}
}

/// Perform a single swap
#[inline]
fn do_swap<M, N, L>(mat: &M, med: &mut Vec<usize>, data: &mut Vec<Rec<N>>, b: usize, j: usize) -> L
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	assert!(b < med.len(), "invalid medoid number");
	assert!(j < n, "invalid object number");
	med[b] = j;
	let mut newloss = L::zero();
	for (o, reco) in data.iter_mut().enumerate() {
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
		let djo = mat.get(j, o);
		// Nearest medoid is gone:
		if reco.near.i == b as u32 {
			if djo < reco.seco.d {
				reco.near = DistancePair {
					i: b as u32,
					d: djo,
				};
			} else {
				reco.near = reco.seco;
				reco.seco = update_second_nearest(mat, med, reco.near.i as usize, b, o, djo);
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
				reco.seco = update_second_nearest(mat, med, reco.near.i as usize, b, o, djo);
			} else if djo < reco.seco.d {
				reco.seco = DistancePair {
					i: b as u32,
					d: djo,
				};
			}
		}
		newloss += L::from(reco.near.d);
	}
	#[cfg(feature = "assertions")]
	debug_validate_assignment(mat, med, &data);
	newloss
}

/// Random initialization (requires the `rand` crate)
///
/// This is simply a call to `rand::seq::index::sample`.
///
/// * `n` - size of the data set
/// * `k` - number of clusters to find
/// * `rng` - random number generator
///
/// returns a vector of medoid indexes in 0..n-1
///
/// ## Example
///
/// Given a dissimilarity matrix of size n x n, use:
/// ```
/// let mut meds = kmedoids::random_initialization(10, 2, &mut rand::thread_rng());
/// println!("Chosen medoids: {:?}", meds);
/// ```
#[cfg(feature = "rand")]
#[inline]
pub fn random_initialization(n: usize, k: usize, mut rng: &mut impl rand::Rng) -> Vec<usize> {
	rand::seq::index::sample(&mut rng, n, k).into_vec()
}

/// Run the FasterPAM algorithm.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
///
/// returns a tuple containing:
/// * the final loss
/// * the final cluster assignment
/// * the number of iterations needed
/// * the number of swaps performed
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
/// * panics when k is 0 or larger than N
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::fasterpam(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn fasterpam<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let mut data = Vec::<Rec<N>>::with_capacity(n);
	let mut loss = initial_assignment(mat, med, &mut data);
	#[cfg(feature = "assertions")]
	debug_validate_assignment(mat, med, &data);

	// println!("Initial loss is {}", loss);
	let mut removal_loss = vec![L::zero(); k];
	update_removal_loss(&data, &mut removal_loss);
	let mut lastswap = n;
	let mut n_swaps = 0;
	let mut iter = 0;
	while iter < maxiter {
		iter += 1;
		// println!("Iteration {} before {}", iter, loss);
		let swaps_before = n_swaps;
		for j in 0..n {
			if j == lastswap {
				break;
			}
			if j == data[j].near.i as usize {
				continue; // This already is a medoid
			}
			let (change, b) = find_best_swap(mat, &removal_loss, &data, j);
			if change >= L::zero() {
				continue; // No improvement
			}
			n_swaps += 1;
			lastswap = j;
			// perform the swap
			let newloss = do_swap(mat, med, &mut data, b, j);
			// println!("{} + {} = {} vs. {}", loss, change, loss + change, newloss);
			if newloss >= loss {
				break; // Probably numerically unstable now.
			}
			loss = newloss;
			update_removal_loss(&data, &mut removal_loss);
		}
		if n_swaps == swaps_before {
			break; // converged
		}
	}
	// println!("final loss: {}", loss);
	// println!("number of swaps: {}", n_swaps);
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, iter, n_swaps)
}

/// Run the FastPAM1 algorithm, which yields the same results as the original PAM.
///
/// This is faster than PAM, but slower than FasterPAM, and mostly of interest for academic reasons.
/// Quality-wise, FasterPAM is not worse on average, but much faster.
///
/// This is the improved version from the journal version of the paper,
/// which costs O(n²) per iteration to find the best swap.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
///
/// returns a tuple containing:
/// * the final loss
/// * the final cluster assignment
/// * the number of iterations needed
/// * the number of swaps performed
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
/// * panics when k is 0 or larger than N
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::fastpam1(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn fastpam1<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let mut data = Vec::<Rec<N>>::with_capacity(n);
	let mut loss = initial_assignment(mat, med, &mut data);
	#[cfg(feature = "assertions")]
	debug_validate_assignment(mat, med, &data);
	// println!("Initial loss is {}", loss);
	let mut removal_loss = vec![L::zero(); k];
	let mut n_swaps = 0;
	let mut iter = 0;
	while iter < maxiter {
		iter += 1;
		// println!("Iteration {} before {}", iter, loss);
		let mut best = (L::zero(), usize::MAX, usize::MAX);
		update_removal_loss(&data, &mut removal_loss);
		for j in 0..n {
			if j == data[j].near.i as usize {
				continue; // This already is a medoid
			}
			let (change, b) = find_best_swap(mat, &removal_loss, &data, j);
			if change >= best.0 {
				continue; // No improvement
			}
			best = (change, b, j);
		}
		if best.0 < L::zero() {
			n_swaps += 1;
			// perform the swap
			let newloss = do_swap(mat, med, &mut data, best.1, best.2);
			// println!("{} + {} = {} vs. {}", loss, best.0, loss + best.0, newloss);
			if newloss >= loss {
				break; // Probably numerically unstable now.
			}
			loss = newloss;
		} else {
			break; // No improvement, or NaN.
		}
	}
	// println!("final loss: {}", loss);
	// println!("number of swaps: {}", n_swaps);
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, iter, n_swaps)
}

/// Main optimization function of PAM, not exposed (use pam_swap or pam)
fn pam_optimize<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	data: &mut Vec<Rec<N>>,
	maxiter: usize,
	mut loss: L,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	#[cfg(feature = "assertions")]
	debug_validate_assignment(mat, med, &data);
	// println!("Initial loss is {}", loss);
	let mut n_swaps = 0;
	let mut iter = 0;
	while iter < maxiter {
		iter += 1;
		// println!("Iteration {} before {}", iter, loss);
		let mut best = (L::zero(), k, usize::MAX);
		for j in 0..n {
			if j == data[j].near.i as usize {
				continue; // This already is a medoid
			}
			let (change, b) = find_best_swap_pam(mat, med, data, j);
			if change >= best.0 {
				continue; // No improvement
			}
			best = (change, b, j);
		}
		if best.0 < L::zero() {
			n_swaps += 1;
			// perform the swap
			let newloss = do_swap(mat, med, data, best.1, best.2);
			// println!("{} + {} = {} vs. {}", loss, best.0, loss + best.0, newloss);
			if newloss >= loss {
				break; // Probably numerically unstable now.
			}
			loss = newloss;
		} else {
			break; // No improvement, or NaN.
		}
	}
	// println!("final loss: {}", loss);
	// println!("number of swaps: {}", n_swaps);
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, iter, n_swaps)
}
/// Not exposed. Use pam_build or pam.
fn pam_build_initialize<M, N, L>(
	mat: &M,
	meds: &mut Vec<usize>,
	data: &mut Vec<Rec<N>>,
	k: usize,
) -> L
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	// choose first medoid
	let mut best = (L::zero(), k);
	for i in 0..n {
		let mut sum = L::zero();
		for j in 0..n {
			if j != i {
				sum += L::from(mat.get(i, j));
			}
		}
		if i == 0 || sum < best.0 {
			best = (sum, i);
		}
	}
	let mut loss = best.0;
	meds.push(best.1);
	for j in 0..n {
		data.push(Rec::<N> {
			near: DistancePair {
				i: 0,
				d: mat.get(best.1, j),
			},
			seco: DistancePair {
				i: u32::MAX,
				d: N::zero(),
			},
		});
	}
	// choose remaining medoids
	for l in 1..k {
		best = (L::zero(), k);
		for (i, _) in data.iter().enumerate().skip(1) {
			let mut sum = L::zero();
			for (j, dj) in data.iter().enumerate() {
				if j != i {
					let d = mat.get(i, j);
					if d < dj.near.d {
						sum += L::from(d) - L::from(dj.near.d)
					}
				}
			}
			if i == 0 || sum < best.0 {
				best = (sum, i);
			}
		}
		assert!(best.0 <= L::zero());
		// Update assignments:
		loss = L::zero();
		for (j, recj) in data.iter_mut().enumerate() {
			if j == best.1 {
				recj.seco = recj.near;
				recj.near = DistancePair {
					i: l as u32,
					d: N::zero(),
				};
				continue;
			}
			let dj = mat.get(best.1, j);
			if dj < recj.near.d {
				recj.seco = recj.near;
				recj.near = DistancePair { i: l as u32, d: dj };
			} else if recj.seco.i == u32::MAX || dj < recj.seco.d {
				recj.seco = DistancePair { i: l as u32, d: dj };
			}
			loss += L::from(recj.near.d);
		}
		meds.push(best.1);
	}
	loss
}

/// Implementation of the original PAM SWAP algorithm (no BUILD).
///
/// This is provided for academic reasons to see the performance difference.
/// Quality-wise, FasterPAM is not worse on average, but much faster.
/// FastPAM1 is supposed to do the same swaps, and find the same result, but faster.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
///
/// returns a tuple containing:
/// * the final loss
/// * the final cluster assignment
/// * the number of iterations needed
/// * the number of swaps performed
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
/// * panics when k is 0 or larger than N
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::pam_swap(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn pam_swap<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let mut data = Vec::<Rec<N>>::with_capacity(mat.len());
	let loss = initial_assignment(mat, med, &mut data);
	pam_optimize(mat, med, &mut data, maxiter, loss)
}

/// Implementation of the original PAM BUILD algorithm.
///
/// This is provided for academic reasons to see the performance difference.
/// Quality-wise, FasterPAM yields better results than just BUILD.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `k` - the number of medoids to pick
///
/// returns a tuple containing:
/// * the initial loss
/// * the initial cluster assignment
/// * the initial medoids
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
/// * panics when k is 0 or larger than N
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let (loss, assi, meds): (f64, _, _) = kmedoids::pam_build(&data, 2);
/// println!("Loss is: {}", loss);
/// ```
pub fn pam_build<M, N, L>(mat: &M, k: usize) -> (L, Vec<usize>, Vec<usize>)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	let mut meds = Vec::<usize>::with_capacity(k);
	let mut data = Vec::<Rec<N>>::with_capacity(n);
	let loss = pam_build_initialize(mat, &mut meds, &mut data, k);
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, meds)
}
/// Implementation of the original PAM algorithm (BUILD + SWAP)
///
/// This is provided for academic reasons to see the performance difference.
/// Quality-wise, FasterPAM is comparable to PAM, and much faster.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `k` - the number of medoids to pick
/// * `maxiter` - the maximum number of iterations allowed
///
/// returns a tuple containing:
/// * the final loss
/// * the final cluster assignment
/// * the final medoids
/// * the number of iterations needed
/// * the number of swaps performed
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
/// * panics when k is 0 or larger than N
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let (loss, assi, meds, n_iter, n_swap): (f64, _, _, _, _) = kmedoids::pam(&data, 2, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn pam<M, N, L>(mat: &M, k: usize, maxiter: usize) -> (L, Vec<usize>, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	let mut meds = Vec::<usize>::with_capacity(k);
	let mut data = Vec::<Rec<N>>::with_capacity(n);
	let loss = pam_build_initialize(mat, &mut meds, &mut data, k);
	let (nloss, assi, n_iter, n_swap) = pam_optimize(mat, &mut meds, &mut data, maxiter, loss);
	(nloss, assi, meds, n_iter, n_swap) // also return medoids
}

/// Assign each to the nearest medoid, return loss
#[inline]
fn assign_nearest<M, N, L>(mat: &M, med: &[usize], data: &mut Vec<usize>) -> L
where
	N: PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	let firstcenter = med[0];
	let mut loss: L = L::zero();
	for i in 0..n {
		let mut best = (0, mat.get(i, firstcenter));
		for (m, &mm) in med.iter().enumerate().skip(1) {
			let dm = mat.get(i, mm);
			if dm < best.1 || i == mm {
				best = (m, dm);
			}
		}
		loss += L::from(best.1);
		assert!(data.len() >= i);
		if data.len() == i {
			data.push(best.0);
		} else {
			data[i] = best.0;
		}
	}
	loss
}

// Choose the best medoid within a partition
pub fn choose_medoid_within_partition<M, N, L>(
	mat: &M,
	assi: &[usize],
	med: &mut [usize],
	m: usize,
) -> (bool, L)
where
	N: PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
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

/// Run the Alternating algorithm, a k-means-style alternate optimization.
///
/// This is fairly fast (also O(n²) as the FasterPAM method), but because the
/// newly chosen medoid must cover the entire existing cluster, it tends to
/// get stuck in worse local optima as the alternatives. Hence, it is not
/// really recommended to use this algorithm (also known as "Alternate" in
/// classic facility location literature, and re-invented by Park and Jun 2009)
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
///
/// returns a tuple containing:
/// * the final loss
/// * the final cluster assignment
/// * the number of iterations needed
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
/// * panics when k is 0 or larger than N
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter): (f64, _, _) = kmedoids::alternating(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn alternating<M, N, L>(mat: &M, med: &mut Vec<usize>, maxiter: usize) -> (L, Vec<usize>, usize)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let mut assi = Vec::<usize>::with_capacity(mat.len());
	let mut loss: L = assign_nearest(mat, med, &mut assi);
	let mut iter = 0;
	while iter < maxiter {
		iter += 1;
		let mut changed = false;
		for i in 0..med.len() {
			changed |= choose_medoid_within_partition::<M, N, L>(mat, &assi, med, i).0;
		}
		if !changed {
			break;
		}
		loss = assign_nearest(mat, med, &mut assi);
	}
	(loss, assi, iter)
}

/// Compute the Silhouette of a strict partitional clustering.
///
/// The Silhouette, proposed by Peter Rousseeuw in 1987, is a popular internal
/// evaluation measure for clusterings. Although it is defined on arbitary metrics,
/// it is most appropriate for evaluating "spherical" clusters, as it expects objects
/// to be closer to all members of its own cluster than to members of other clusters.
///
/// Because of the additional requirement of a division operator, this implementation
/// currently always returns a f64 result, and accepts only input distances that can be
/// converted into f64.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `f64` for the cost (use a float type)
/// * `mat` - a pairwise distance matrix
/// * `assi` - the cluster assignment
/// * `samples` - whether to keep the individual samples, or not
///
/// returns a tuple containing:
/// * the average silhouette
/// * the individual silhouette values (empty if `samples = false`)
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter): (f64, _, _) = kmedoids::alternating(&data, &mut meds, 100);
/// let (sil, _): (f64, _) = kmedoids::silhouette(&data, &assi, false);
/// println!("Silhouette is: {}", sil);
/// ```
pub fn silhouette<M, N, L>(mat: &M, assi: &[usize], samples: bool) -> (L, Vec<L>)
where
	N: Zero + PartialOrd + Copy + std::fmt::Display,
	L: AddAssign
		+ Div<Output = L>
		+ Sub<Output = L>
		+ Signed
		+ Zero
		+ PartialOrd
		+ Copy
		+ From<N>
		+ From<u32>,
	M: ArrayAdapter<N>,
{
	fn checked_div<L>(x: L, y: L) -> L
	where
		L: Div<Output = L> + Zero + Copy + PartialOrd,
	{
		if y > L::zero() {
			x.div(y)
		} else {
			L::zero()
		}
	}
	let mut sil = if samples {
		vec![L::zero(); assi.len()]
	} else {
		vec![L::zero(); 0]
	};
	let mut lsum: L = L::zero();
	let mut buf = Vec::<(u32, L)>::new();
	for (i, &ai) in assi.iter().enumerate() {
		buf.clear();
		for (j, &aj) in assi.iter().enumerate() {
			while aj >= buf.len() {
				buf.push((0, L::zero()));
			}
			if i != j {
				buf[aj].0 += 1;
				buf[aj].1 += mat.get(i, j).into();
			}
		}
		if buf.len() == 1 {
			return (L::zero(), sil);
		}
		let a = checked_div(buf[ai].1, buf[ai].0.into());
		let mut tmp = buf
			.iter()
			.enumerate()
			.filter(|&(i, _)| i != ai)
			.map(|(_, p)| checked_div(p.1, p.0.into()));
		// Ugly hack to get the min():
		let tmp2 = tmp.next().unwrap_or_else(L::zero);
		let b = tmp.fold(tmp2, |x, y| if y < x { x } else { y });
		let s = checked_div(b - a, if a > b { a } else { b });
		if samples {
			sil[i] = s;
		}
		lsum += s;
	}
	if samples {
		assert_eq!(sil.len(), assi.len(), "Length not as expected.");
	}
	(lsum.div((assi.len() as u32).into()), sil)
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{
		alternating, arrayadapter::LowerTriangle, fasterpam, fastpam1, pam, pam_build, pam_swap,
		silhouette,
	};
	fn assert_array(result: Vec<usize>, expect: Vec<usize>, msg: &'static str) {
		assert!(
			result.iter().zip(expect.iter()).all(|(a, b)| a == b),
			"{}",
			msg
		);
	}

	#[test]
	fn testfasterpam_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = fasterpam(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 2, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn testfasterpam_single_cluster() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![1]; // So we need one swap
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = fasterpam(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 14, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 1, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 0, 0], "assignment not as expected");
		assert_array(meds, vec![0], "medoids not as expected");
		assert_eq!(sil, 0., "Silhouette not as expected");
	}

	#[test]
	fn testfastpam_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = fastpam1(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn testpam_swap_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = pam_swap(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn testpam_build_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let (loss, assi, meds): (i64, _, _) = pam_build(&data, 2);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 4, "loss not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn testpam_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let (loss, assi, meds, n_iter, n_swap): (i64, _, _, _, _) = pam(&data, 2, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		// no swaps, because BUILD does a decent job
		assert_eq!(n_swap, 0, "swaps not as expected");
		assert_eq!(n_iter, 1, "iterations not as expected");
		assert_eq!(loss, 4, "loss not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn testalternating() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter): (i64, _, _) = alternating(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(n_iter, 3, "iterations not as expected");
		assert_eq!(loss, 4, "loss not as expected");
		assert_array(assi, vec![1, 1, 1, 0, 0], "assignment not as expected");
		assert_array(meds, vec![3, 0], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}
}
