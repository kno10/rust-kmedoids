#![allow(clippy::needless_range_loop)]
//! k-Medoids Clustering with the FasterPAM Algorithm
//!
//! For details on the implemented FasterPAM algorithm, please see:
//!
//! Erich Schubert, Peter J. Rousseeuw  
//! **Fast and Eager k-Medoids Clustering:  
//! O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms**  
//! Under review at Information Systems, Elsevier.  
//! Preprint: <https://arxiv.org/abs/2008.05171>
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
//! let (loss, assi, n_iter, n_swap) = kmedoids::fasterpam(&data, &mut meds, 100);
//! println!("Loss is: {}", loss);
//! ```
pub mod arrayadapter;
pub mod safeadd;

#[cfg(test)]
#[cfg(bench)]
pub mod bench;

pub use crate::arrayadapter::ArrayAdapter;
pub use crate::safeadd::SafeAdd;
use num_traits::{NumAssignOps, Signed, Zero};

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
fn initial_assignment<M, N>(mat: &M, med: &[usize], data: &mut Vec<Rec<N>>) -> N
where
	N: NumAssignOps + Zero + PartialOrd + Copy + SafeAdd,
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
	let mut loss: N = N::zero();
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
		for m in 1..k {
			let dm = mat.get(i, med[m]);
			if dm < cur.near.d || i == med[m] {
				cur.seco = cur.near;
				cur.near = DistancePair { i: m as u32, d: dm };
			} else if cur.seco.i == u32::MAX || dm < cur.seco.d {
				cur.seco = DistancePair { i: m as u32, d: dm };
			}
		}
		loss.safe_inc(cur.near.d);
		data.push(cur);
	}
	loss
}

/// Update the loss when removing each medoid
fn update_removal_loss<N>(data: &[Rec<N>], loss: &mut Vec<N>)
where
	N: NumAssignOps + Signed + Copy + Zero + SafeAdd,
{
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
fn find_min<N>(a: &[N]) -> (usize, N)
where
	N: PartialOrd + Copy + Zero,
{
	let mut rk: usize = a.len();
	let mut rv: N = N::zero();
	for (ik, iv) in a.iter().enumerate() {
		if ik == 0 || *iv < rv {
			rk = ik;
			rv = *iv;
		}
	}
	(rk, rv)
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
	N: NumAssignOps + PartialOrd + Copy + SafeAdd,
	M: ArrayAdapter<N>,
{
	let mut s = DistancePair {
		i: b as u32,
		d: djo,
	};
	for i in 0..med.len() {
		if i == n || i == b {
			continue;
		}
		let dm = mat.get(o, med[i]);
		if dm < s.d {
			s = DistancePair { i: i as u32, d: dm };
		}
	}
	s
}
/// Find the best swap for object j - FastPAM version
#[inline]
fn find_best_swap<M, N>(mat: &M, removal_loss: &[N], data: &[Rec<N>], j: usize) -> (N, usize)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let mut ploss = removal_loss.to_vec();
	// Improvement from the journal version:
	let mut acc = N::zero();
	for o in 0..n {
		let reco = &data[o];
		let djo = mat.get(j, o);
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
	(bloss + acc, b) // add the shared accumulator
}

/// Find the best swap for object j - slower PAM version
#[inline]
fn find_best_swap_pam<M, N>(mat: &M, med: &[usize], data: &[Rec<N>], j: usize) -> (N, usize)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	let recj = &data[j];
	let mut best = (N::zero(), usize::MAX);
	for m in 0..k {
		let mut acc = -recj.near.d; // j becomes medoid
		for o in 0..n {
			if o == j {
				continue;
			}
			let reco = &data[o];
			let djo = mat.get(j, o);
			// Current medoid is being replaced:
			if reco.near.i as usize == m {
				if djo < reco.seco.d {
					// Assign to new medoid:
					acc.safe_inc(djo - reco.near.d);
				} else {
					// Assign to second nearest instead:
					acc.safe_inc(reco.seco.d - reco.near.d);
				}
			} else if djo < reco.near.d {
				// new mediod is closer:
				acc.safe_inc(djo - reco.near.d);
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
	N: NumAssignOps + PartialOrd + Copy + SafeAdd,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	for o in 0..n {
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
fn do_swap<M, N>(mat: &M, med: &mut Vec<usize>, data: &mut Vec<Rec<N>>, b: usize, j: usize) -> N
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	assert!(b < med.len(), "invalid medoid number");
	assert!(j < n, "invalid object number");
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
				reco.seco = update_second_nearest(mat, &med, reco.near.i as usize, b, o, djo);
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
				reco.seco = update_second_nearest(mat, &med, reco.near.i as usize, b, o, djo);
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
/// returns a vector of medoid indexes in 0..n
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
/// * type `N` - number data type such as `i32` or `f64` (must be signed)
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
/// let (loss, assi, n_iter, n_swap) = kmedoids::fasterpam(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn fasterpam<M, N>(
	mat: &M,
	mut med: &mut Vec<usize>,
	maxiter: usize,
) -> (N, Vec<usize>, usize, usize)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	let mut data = Vec::<Rec<N>>::with_capacity(n);
	let mut loss = initial_assignment(mat, &med, &mut data);
	#[cfg(feature = "assertions")]
	debug_validate_assignment(&mat, &med, &data);

	// println!("Initial loss is {}", loss);
	let mut removal_loss = vec![N::zero(); k];
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
			if change >= N::zero() {
				continue; // No improvement
			}
			n_swaps += 1;
			lastswap = j;
			// perform the swap
			let newloss = do_swap(mat, &mut med, &mut data, b, j);
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
/// * type `N` - number data type such as `i32` or `f64` (must be signed)
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
/// let (loss, assi, n_iter, n_swap) = kmedoids::fastpam1(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn fastpam1<M, N>(
	mat: &M,
	mut med: &mut Vec<usize>,
	maxiter: usize,
) -> (N, Vec<usize>, usize, usize)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	let mut data = Vec::<Rec<N>>::with_capacity(n);
	let mut loss = initial_assignment(mat, &med, &mut data);
	#[cfg(feature = "assertions")]
	debug_validate_assignment(&mat, &med, &data);
	// println!("Initial loss is {}", loss);
	let mut removal_loss = vec![N::zero(); k];
	let mut n_swaps = 0;
	let mut iter = 0;
	while iter < maxiter {
		iter += 1;
		// println!("Iteration {} before {}", iter, loss);
		let mut best = (N::zero(), usize::MAX, usize::MAX);
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
		if best.0 < N::zero() {
			n_swaps += 1;
			// perform the swap
			let newloss = do_swap(mat, &mut med, &mut data, best.1, best.2);
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
fn pam_optimize<M, N>(
	mat: &M,
	mut med: &mut Vec<usize>,
	mut data: &mut Vec<Rec<N>>,
	maxiter: usize,
	mut loss: N,
) -> (N, Vec<usize>, usize, usize)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	#[cfg(feature = "assertions")]
	debug_validate_assignment(&mat, &med, &data);
	// println!("Initial loss is {}", loss);
	let mut n_swaps = 0;
	let mut iter = 0;
	while iter < maxiter {
		iter += 1;
		// println!("Iteration {} before {}", iter, loss);
		let mut best = (N::zero(), k, usize::MAX);
		for j in 0..n {
			if j == data[j].near.i as usize {
				continue; // This already is a medoid
			}
			let (change, b) = find_best_swap_pam(mat, &med, &data, j);
			if change >= best.0 {
				continue; // No improvement
			}
			best = (change, b, j);
		}
		if best.0 < N::zero() {
			n_swaps += 1;
			// perform the swap
			let newloss = do_swap(mat, &mut med, &mut data, best.1, best.2);
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
fn pam_build_initialize<M, N>(mat: &M, meds: &mut Vec<usize>, data: &mut Vec<Rec<N>>, k: usize) -> N
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	// choose first medoid
	let mut best = (N::zero(), k);
	for i in 0..n {
		let mut sum = N::zero();
		for j in 0..n {
			if j != i {
				sum += mat.get(i, j);
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
		best = (N::zero(), k);
		for i in 1..n {
			let mut sum = N::zero();
			for j in 0..n {
				if j != i {
					let d = mat.get(i, j);
					if d < data[j].near.d {
						sum += d - data[j].near.d;
					}
				}
			}
			if i == 0 || sum < best.0 {
				best = (sum, i);
			}
		}
		assert!(best.0 <= N::zero());
		// Update assignments:
		loss = N::zero();
		for j in 0..n {
			let mut recj = &mut data[j];
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
			loss.safe_inc(recj.near.d);
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
/// * type `N` - number data type such as `i32` or `f64` (must be signed)
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
/// let (loss, assi, n_iter, n_swap) = kmedoids::pam_swap(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn pam_swap<M, N>(
	mat: &M,
	mut med: &mut Vec<usize>,
	maxiter: usize,
) -> (N, Vec<usize>, usize, usize)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd + std::fmt::Display,
	M: ArrayAdapter<N>,
{
	let mut data = Vec::<Rec<N>>::with_capacity(mat.len());
	let loss = initial_assignment(mat, &med, &mut data);
	pam_optimize(mat, &mut med, &mut data, maxiter, loss)
}

/// Implementation of the original PAM BUILD algorithm.
///
/// This is provided for academic reasons to see the performance difference.
/// Quality-wise, FasterPAM yields better results than just BUILD.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `i32` or `f64` (must be signed)
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
/// let (loss, assi, meds) = kmedoids::pam_build(&data, 2);
/// println!("Loss is: {}", loss);
/// ```
pub fn pam_build<M, N>(mat: &M, k: usize) -> (N, Vec<usize>, Vec<usize>)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd + std::fmt::Display,
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
/// * type `N` - number data type such as `i32` or `f64` (must be signed)
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
/// let (loss, assi, meds, n_iter, n_swap) = kmedoids::pam(&data, 2, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn pam<M, N>(mat: &M, k: usize, maxiter: usize) -> (N, Vec<usize>, Vec<usize>, usize, usize)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd + std::fmt::Display,
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
	for o in data.iter() {
		println!("{} {} {} {}", o.near.i, o.near.d, o.seco.i, o.seco.d);
	}
	let (nloss, assi, n_iter, n_swap) = pam_optimize(mat, &mut meds, &mut data, maxiter, loss);
	(nloss, assi, meds, n_iter, n_swap) // also return medoids
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.

	use crate::{arrayadapter::LowerTriangle, fasterpam, fastpam1, pam, pam_build, pam_swap};
	fn assert_array(result: Vec<usize>, expect: Vec<usize>, msg: &'static str) {
		assert!(result.iter().zip(expect.iter()).all(|(a, b)| a == b), msg);
	}

	#[test]
	fn testfasterpam_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap) = fasterpam(&data, &mut meds, 10);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 2, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
	}

	#[test]
	fn testfastpam_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap) = fastpam1(&data, &mut meds, 10);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
	}

	#[test]
	fn testpam_swap_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap) = pam_swap(&data, &mut meds, 10);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
	}

	#[test]
	fn testpam_build_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let (loss, assi, meds) = pam_build(&data, 2);
		assert_eq!(loss, 4, "loss not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
	}

	#[test]
	fn testpam_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let (loss, assi, meds, n_iter, n_swap) = pam(&data, 2, 10);
		// no swaps, because BUILD does a decent job
		assert_eq!(n_swap, 0, "swaps not as expected");
		assert_eq!(n_iter, 1, "iterations not as expected");
		assert_eq!(loss, 4, "loss not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
	}
}
