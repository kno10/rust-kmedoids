#[cfg(feature = "parallel")]
use crate::arrayadapter::ArrayAdapter;
use crate::fasterpam::{update_removal_loss, update_second_nearest};
use crate::util::*;
use core::ops::AddAssign;
use ndarray::Array;
use num_traits::{Signed, Zero};
use rayon::prelude::*;
use std::convert::From;
use std::sync::{Arc, Mutex};

/// Run the FasterPAM algorithm (parallel version).
///
/// For small data sets (n<1000) it is usually faster to use the non-parallel version.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
/// * `rng` - random number generator for shuffling the input data
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
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::par_fasterpam(&data, &mut meds, 100, &mut rand::thread_rng());
/// println!("Loss is: {}", loss);
/// ```
pub fn par_fasterpam<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
	rng: &mut impl rand::Rng,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Copy + Sync + Send,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + Sync + Send,
	M: ArrayAdapter<N> + Sync + Send,
{
	let (n, k) = (mat.len(), med.len());
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let (mut loss, mut data) = par_initial_assignment(mat, med);
	debug_assert_assignment(mat, med, &data);

	let mut removal_loss = vec![L::zero(); k];
	update_removal_loss(&data, &mut removal_loss);
	let (mut lastswap, mut n_swaps, mut iter) = (n, 0, 0);
	let seq = rand::seq::index::sample(rng, n, n); // random shuffling
	while iter < maxiter {
		iter += 1;
		let swaps_before = n_swaps;
		for j in seq.iter() {
			if j == lastswap {
				break;
			}
			if j == med[data[j].near.i as usize] {
				continue; // This already is a medoid
			}
			let (change, b) = par_find_best_swap(mat, &removal_loss, &data, j);
			if change >= L::zero() {
				continue; // No improvement
			}
			n_swaps += 1;
			lastswap = j;
			// perform the swap
			let newloss = par_do_swap(mat, med, &mut data, b, j);
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
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, iter, n_swaps)
}

/// Perform the initial assignment to medoids
#[inline]
fn par_initial_assignment<M, N, L>(mat: &M, med: &[usize]) -> (L, Vec<Rec<N>>)
where
	N: Zero + PartialOrd + Copy + Send + Sync,
	L: AddAssign + Zero + PartialOrd + Copy + From<N> + Send + Sync,
	M: ArrayAdapter<N> + Sync,
{
	let n = mat.len();
	let k = med.len();
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	let mut data = vec![Rec::<N>::empty(); mat.len()];
	let firstcenter = med[0];
	let loss = data
		.par_iter_mut()
		.enumerate()
		.map(|(i, cur)| {
			*cur = Rec::new(0, mat.get(i, firstcenter), u32::MAX, N::zero());
			for (m, &me) in med.iter().enumerate().skip(1) {
				let d = mat.get(i, me);
				if d < cur.near.d || i == me {
					cur.seco = cur.near;
					cur.near = DistancePair { i: m as u32, d };
				} else if cur.seco.i == u32::MAX || d < cur.seco.d {
					cur.seco = DistancePair { i: m as u32, d };
				}
			}
			L::from(cur.near.d)
		})
		.reduce_with(L::add)
		.unwrap();
	(loss, data)
}

/// Find the best swap for object j - FastPAM version (parallel version - without shared data)
#[inline]
fn par_find_best_swap<M, N, L>(mat: &M, removal_loss: &[L], data: &[Rec<N>], j: usize) -> (L, usize)
where
	N: Zero + PartialOrd + Copy + Sync + Send,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + Sync + Send,
	M: ArrayAdapter<N> + Sync + Send,
{
	let n = mat.len();
	let length: usize = removal_loss.len();
	let mut ploss = Array::from_vec(removal_loss.to_vec());
	let mut acc = L::zero();
	rayon::scope(|s| {
		let parts = rayon::current_num_threads();
		let stepsize = (n + parts - 1) / parts; // upper division
		let mutex = Arc::new(Mutex::new((&mut ploss, &mut acc)));
		for x in 0..parts {
			let mutex = Arc::clone(&mutex);
			s.spawn(move |_| {
				let mut loss = Array::zeros(length);
				let mut lagg = L::zero();
				let start = x * stepsize;
				let end = usize::min(start + stepsize, n);
				for o in start..end {
					let reco = &data[o];
					let djo = mat.get(j, o);
					// New medoid is closest:
					if djo < reco.near.d {
						lagg += L::from(djo) - L::from(reco.near.d);
						// loss already includes ds - dn, remove
						loss[reco.near.i as usize] += L::from(reco.near.d) - L::from(reco.seco.d);
					} else if djo < reco.seco.d {
						// loss already includes ds - dn, adjust to d(xo) - dn
						loss[reco.near.i as usize] += L::from(djo) - L::from(reco.seco.d);
					}
				}
				// Synchronize for merging the results
				let mut mutex = mutex.lock().unwrap();
				*mutex.0 += &loss;
				*mutex.1 += lagg;
			})
		}
	});
	let (b, bloss): (usize, L) = find_min(&mut ploss.iter());
	(bloss + acc, b) // add the shared accumulator
}

/// Perform a single swap -- parallel version
#[inline]
fn par_do_swap<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	data: &mut Vec<Rec<N>>,
	b: usize,
	j: usize,
) -> L
where
	N: Zero + PartialOrd + Copy + Send + Sync,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + Send + Sync,
	M: ArrayAdapter<N> + Sync,
{
	let n = mat.len();
	assert!(b < med.len(), "invalid medoid number");
	assert!(j < n, "invalid object number");
	med[b] = j;
	data.par_iter_mut()
		.enumerate()
		.map(|(o, reco)| {
			if o == j {
				if reco.near.i != b as u32 {
					reco.seco = reco.near;
				}
				reco.near = DistancePair::new(b as u32, N::zero());
				return L::zero();
			}
			let djo = mat.get(j, o);
			// Nearest medoid is gone:
			if reco.near.i == b as u32 {
				if djo < reco.seco.d {
					reco.near = DistancePair::new(b as u32, djo);
				} else {
					reco.near = reco.seco;
					reco.seco = update_second_nearest(mat, med, reco.near.i as usize, b, o, djo);
				}
			} else {
				// nearest not removed
				if djo < reco.near.d {
					reco.seco = reco.near;
					reco.near = DistancePair::new(b as u32, djo);
				} else if reco.seco.i == b as u32 {
					// second nearest was replaced
					reco.seco = update_second_nearest(mat, med, reco.near.i as usize, b, o, djo);
				} else if djo < reco.seco.d {
					reco.seco = DistancePair::new(b as u32, djo);
				}
			}
			L::from(reco.near.d)
		})
		.reduce_with(L::add)
		.unwrap()
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{arrayadapter::LowerTriangle, par_fasterpam, par_silhouette, util::assert_array};
	use rand::{rngs::StdRng, SeedableRng};

	#[test]
	fn test_fasterpam_par() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let mut rng = StdRng::seed_from_u64(1);
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) =
			par_fasterpam(&data, &mut meds, 10, &mut rng);
		let sil: f64 = par_silhouette(&data, &assi);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 4], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}
}
