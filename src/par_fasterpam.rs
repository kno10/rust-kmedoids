use crate::arrayadapter::ArrayAdapter;
use crate::fasterpam::{do_swap, initial_assignment, update_removal_loss};
use crate::util::*;
use core::ops::AddAssign;
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use num_traits::{Signed, Zero};
use rayon::prelude::*;
use std::convert::From;

/// Run the FasterPAM algorithm (parallel version).
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `maxiter` - the maximum number of iterations allowed
/// * `threads` - number of threads for rayon
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
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::par_fasterpam(&data, &mut meds, 100, 2);
/// println!("Loss is: {}", loss);
/// ```
pub fn par_fasterpam<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
	threads: usize,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Copy + Sync + Send,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + Sync + Send,
	M: ArrayAdapter<N> + Sync + Send,
{
	let n = mat.len();
	let k = med.len();
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let (mut loss, mut data) = initial_assignment(mat, med);
	debug_assert_assignment(mat, med, &data);
	let mut threads_used = threads;
	// set number of threads
	if threads > 0 {
		rayon::ThreadPoolBuilder::new()
			.num_threads(threads)
			.build_global()
			.unwrap();
	} else {
		threads_used = num_cpus::get();
	}

	let mut removal_loss = vec![L::zero(); k];
	update_removal_loss(&data, &mut removal_loss);
	let mut lastswap = n;
	let mut n_swaps = 0;
	let mut iter = 0;
	while iter < maxiter {
		iter += 1;
		let swaps_before = n_swaps;
		for j in 0..n {
			if j == lastswap {
				break;
			}
			if j == data[j].near.i as usize {
				continue; // This already is a medoid
			}
			let (change, b) = find_best_swap_par(mat, &removal_loss, &data, j, threads_used);
			if change >= L::zero() {
				continue; // No improvement
			}
			n_swaps += 1;
			lastswap = j;
			// perform the swap
			let newloss = do_swap(mat, med, &mut data, b, j);
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

/// Find the best swap for object j - FastPAM version (parallel version - without shared data)
#[inline]
fn find_best_swap_par<M, N, L>(
	mat: &M,
	removal_loss: &[L],
	data: &[Rec<N>],
	j: usize,
	threads: usize,
) -> (L, usize)
where
	N: Zero + PartialOrd + Copy + Sync + Send,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + Sync + Send,
	M: ArrayAdapter<N> + Sync + Send,
{
	let n = mat.len();
	let stepsize = n / threads;
	let length: usize = removal_loss.len();
	let mut init_ploss = removal_loss.to_vec();
	init_ploss.push(L::zero());
	let mut ploss = Array::from_vec(init_ploss);
	(0..threads)
		.into_par_iter()
		.map(|x| {
			let mut loss = Array::zeros(length + 1);
			let start = x * stepsize;
			let mut end = (x + 1) * stepsize;
			if x == threads {
				end = n;
			}
			for o in start..end {
				let reco = &data[o];
				let djo = mat.get(j, o);
				// New medoid is closest:
				if djo < reco.near.d {
					loss[length] += L::from(djo) - L::from(reco.near.d);
					// loss already includes ds - dn, remove
					loss[reco.near.i as usize] += L::from(reco.near.d) - L::from(reco.seco.d);
				} else if djo < reco.seco.d {
					// loss already includes ds - dn, adjust to d(xo) - dn
					loss[reco.near.i as usize] += L::from(djo) - L::from(reco.seco.d);
				}
			}
			loss
		})
		.collect::<Vec<ArrayBase<OwnedRepr<L>, Dim<[usize; 1]>>>>()
		.iter()
		.for_each(|x| ploss += x);
	find_min_acc(&ploss.to_vec())
}

/// Find the minimum (both index and value+acc)
#[inline]
fn find_min_acc<L>(a: &[L]) -> (L, usize)
where
	L: PartialOrd + Copy + Zero + AddAssign,
{
	let mut rk: usize = a.len();
	let mut rv: L = L::zero();
	for (ik, iv) in a.iter().enumerate() {
		if ik == (a.len() - 1) {
			rv += *iv;
			continue;
		}
		if ik == 0 || *iv < rv {
			rk = ik;
			rv = *iv;
		}
	}
	(rv, rk)
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{arrayadapter::LowerTriangle, par_fasterpam, silhouette, util::assert_array};

	#[test]
	fn test_fasterpam_par() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = par_fasterpam(&data, &mut meds, 10, 2);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 2, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}
}
