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
//!
//! If you use this in scientific work, please consider citing above articles.
//!
//! ## Example
//!
//! Given a dissimilarity matrix of size 4 x 4, use:
//! ```
//! let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
//! let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
//! let (loss, numswap, numiter, assignment) = kmedoids::fasterpam(&data, &mut meds, 100);
//! println!("Loss is: {}", loss);
//! ```
pub mod arrayadapter;
pub mod safeadd;

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
fn initial_assignment<M, N>(mat: &M, med: &Vec<usize>, data: &mut Vec<Rec<N>>) -> N
where
	N: NumAssignOps + Zero + PartialOrd + Copy + SafeAdd,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	let firstcenter = med[0];
	let unassigned = k as u32;
	let mut loss: N = N::zero();
	for i in 0..n {
		let mut cur = Rec::<N> {
			near: DistancePair {
				i: 0,
				d: mat.get(i, firstcenter),
			},
			seco: DistancePair {
				i: unassigned,
				d: N::zero(),
			},
		};
		for m in 1..k {
			let d = mat.get(i, med[m]);
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
fn update_removal_loss<N>(data: &Vec<Rec<N>>, loss: &mut Vec<N>)
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
fn find_min<N>(a: &Vec<N>) -> (usize, N)
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
	return (rk, rv);
}

/// Update the second nearest medoid information
///
/// Called after each swap.
#[inline]
fn update_second_nearest<M, N>(
	mat: &M,
	med: &Vec<usize>,
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
	return s;
}
/// Find the best swap
#[inline]
fn find_best_swap<M, N>(mat: &M, removal_loss: &Vec<N>, data: &Vec<Rec<N>>, j: usize) -> (usize, N)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let mut ploss = removal_loss.clone();
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
	return (b, bloss + acc);
}

/// Debug helper function
#[cfg(feature = "assertions")]
fn debug_validate_assignment<M, N>(mat: &M, med: &Vec<usize>, data: &Vec<Rec<N>>)
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
	return newloss;
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
pub fn random_initialization(n: usize, k: usize, mut rng: &mut impl rand::Rng) -> Vec<usize> {
	return rand::seq::index::sample(&mut rng, n, k).into_vec();
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
/// * the number of swaps performed
/// * the number of iterations needed
/// * the final cluster assignment
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
/// let (loss, numswap, numiter, assignment) = kmedoids::fasterpam(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn fasterpam<M, N>(
	mat: &M,
	mut med: &mut Vec<usize>,
	maxiter: usize,
) -> (N, usize, usize, Vec<usize>)
where
	N: NumAssignOps + Signed + Zero + PartialOrd + Copy + SafeAdd,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	let mut data = Vec::<Rec<N>>::with_capacity(n);
	let mut loss = initial_assignment(mat, &med, &mut data);
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
			let (b, change) = find_best_swap(mat, &removal_loss, &data, j);
			if change >= N::zero() {
				continue; // No improvement
			}
			numswaps += 1;
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
		if numswaps == swaps_before {
			break; // converged
		}
	}
	// println!("final loss: {}", loss);
	// println!("number of swaps: {}", numswaps);
	let assignment = data.iter().map(|x| x.near.i as usize).collect();
	return (loss, numswaps, iter, assignment);
}

#[cfg(test)]
mod tests {
	use crate::arrayadapter::LowerTriangle;
	use crate::fasterpam;
	#[test]
	fn basic() {
		// TODO: use a larger, much more interesting example.
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, numswap, numiter, assignment) = fasterpam(&data, &mut meds, 10);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(numswap, 2, "swaps not as expected");
		assert_eq!(numiter, 2, "iterations not as expected");
		let expected = vec![0, 0, 0, 1, 1];
		assert!(
			assignment.iter().zip(expected.iter()).all(|(a, b)| a == b),
			"cluster assignment not as expected"
		);
	}
}
