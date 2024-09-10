use crate::arrayadapter::ArrayAdapter;
use crate::util::*;
use core::ops::AddAssign;
use num_traits::{Signed, Zero, Float};
use std::convert::From;
use std::cmp::{min, max};
use crate::fastermsc::{initial_assignment,update_removal_loss,find_best_swap,do_swap,fastermsc_k2};

#[inline]
fn _loss<N, L>(a: N, b: N) -> L
	where
		N: Zero,
		L: Float + From<N>,
{
	if N::is_zero(&a) || N::is_zero(&b) { L::zero() } else { <L as From<N>>::from(a) / <L as From<N>>::from(b) } 
}

/// Run the DynMSC algorithm.
///
/// We begin with a maximum number of clusters, optimize the Average Medoid Silhouette,
/// then decrease the number of clusters by one,
/// and repeat until we have reached a minimum number of clusters.
/// During this process, we store the solution with the highest AMS to return later.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `i64` or `f64` for the loss (must be signed)
/// * `mat` - a pairwise distance matrix
/// * `med` - the list of medoids
/// * `mink` - the minimum number of clusters
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
/// let (loss, assi, n_iter, n_swap, meds, losses): (f64, _, _, _, _, _) = kmedoids::dynmsc(&data, &meds, 2, 100);
/// println!("Loss is: {}", loss);
/// println!("Best k: {}", meds.len());
/// ```
pub fn dynmsc<M, N, L>(
	mat: &M,
	med: &[usize],
	mink: usize,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize, Vec<usize>, Vec<L>)
	where
		N: Zero + PartialOrd + Clone,
		L: Float + Signed + AddAssign + From<N> + From<u32> + std::fmt::Debug,
		M: ArrayAdapter<N>,
{
	let mut med = med.to_owned();
	let (n, mut k) = (mat.len(), med.len());
	let minimum_k = min(max(mink, 1), k);
	if k == 1 {
		let mut return_loss = vec![L::zero(); 1_usize];
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, &mut med, 0);
		return_loss[0] = loss;
		let return_meds = med.clone();
		return (loss, assi, 1, if swapped { 1 } else { 0 }, return_meds, return_loss);
	}
	let (mut loss, mut data): (L, _) = initial_assignment(mat, &med);
	debug_assert_assignment_th(mat, &med, &data);

	let mut return_loss = vec![L::zero(); k - minimum_k + 1];
	let mut best_loss = L::zero();
	let mut return_assi = vec![0, n];
	let mut return_iter = 0;
	let mut return_swaps = 0;
	let (mut lastswap, mut n_swaps, mut iter);
	let mut removal_loss = vec![L::zero(); k];
	let mut return_meds = med.clone();
	while k >= 3 && k >= minimum_k {
		update_removal_loss(&data, &mut removal_loss);
		lastswap = n;
		n_swaps = 0;
		iter = 0;
		while iter < maxiter {
			iter += 1;
			let (swaps_before, lastloss) = (n_swaps, loss);
			for j in 0..n {
				if j == lastswap {
					break;
				}
				if j == med[data[j].near.i as usize] {
					continue; // This already is a medoid
				}
				let (change, b) = find_best_swap(mat, &removal_loss, &data, j);
				if change <= L::zero() {
					continue; // No improvement
				}
				n_swaps += 1;
				lastswap = j;
				// perform the swap
				loss = do_swap(mat, &mut med, &mut data, b, j);
				update_removal_loss(&data, &mut removal_loss);
			}
			if n_swaps == swaps_before || loss >= lastloss {
				break; // converged
			}
		}
		let mut r = (L::zero(), usize::MAX);
		for (o, remloss) in removal_loss.iter().enumerate() {
			if o == 0 || *remloss < r.0 {
				r = (*remloss, o);
			}
		}
		loss = L::one() - loss / <L as From<u32>>::from(n as u32);
		return_loss[k - minimum_k] = loss;
		let assi = data.iter().map(|x| x.near.i as usize).collect();
		if loss > best_loss {
			best_loss = loss;
			return_assi = assi;
			return_meds = med.clone();
		}
		return_swaps += n_swaps;
		return_iter += iter;
		loss = remove_med(mat, &mut med, &mut data, r.1);
		removal_loss.remove(r.1);
		k = med.len();
	}
	if minimum_k <= 2 {
		let (loss2, assi2, iter2, n_swaps2): (L, _, _, _) = fastermsc_k2(mat, &mut med, maxiter);
		return_loss[2 - minimum_k] = loss2;
		if loss2 > best_loss {
			best_loss = loss2;
			return_meds = med.clone();
			return_assi = assi2;
		}
		return_swaps += n_swaps2;
		return_iter += iter2;
	}
	if minimum_k <= 1 {
		return_loss[0] = L::zero();
	}
	(best_loss, return_assi, return_iter, return_swaps, return_meds, return_loss)
}

/// Update the third nearest medoid information
/// Called after each swap.
#[inline]
pub(crate) fn update_third_nearest_without_new<M, N>(
	mat: &M,
	med: &[usize],
	n: usize,
	s: usize,
	b: usize,
	o: usize,
) -> DistancePair<N>
	where
		N: Zero + PartialOrd + Clone,
		M: ArrayAdapter<N>,
{
	let mut dist = DistancePair::new(b as u32, N::zero());
	for (i, &mi) in med.iter().enumerate() {
		if i == n || i == s {
			continue;
		}
		let d = mat.get(o, mi);
		if dist.i == (b as u32) || d < dist.d {
			dist = DistancePair::new(i as u32, d);
		}
	}
	dist
}

/// Remove one medoid
#[inline]
pub(crate) fn remove_med<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	data: &mut [Reco<N>],
	b: usize,
) -> L
	where
		N: Zero + PartialOrd + Clone,
		L: Float + Signed + AddAssign + From<N>,
		M: ArrayAdapter<N>,
{
	let l= med.len() - 1;
	assert!(b < med.len(), "invalid medoid number");
	med[b] = med[l];
	med.remove(l);
	data.iter_mut()
		.enumerate()
		.map(|(o, reco)| {
			if reco.near.i == b as u32 {
				// nearest medoid is gone
				if reco.seco.i == l as u32 {
					reco.seco.i = b as u32;
				}
				if reco.third.i == l as u32 {
					reco.third.i = b as u32;
				}
				reco.near = reco.seco.clone();
				reco.seco = reco.third.clone();
				reco.third = update_third_nearest_without_new(mat, med, reco.near.i as usize, reco.seco.i as usize, b, o);
			} else if reco.seco.i == b as u32 {
				// second nearest is gone
				if reco.near.i == l as u32 {
					reco.near.i = b as u32;
				}
				if reco.third.i == l as u32 {
					reco.third.i = b as u32;
				}
				reco.seco = reco.third.clone();
				reco.third = update_third_nearest_without_new(mat, med, reco.near.i as usize, reco.seco.i as usize, b, o);
			} else if reco.third.i == b as u32 {
				// third nearest is gone
				if reco.near.i == l as u32 {
					reco.near.i = b as u32;
				}
				if reco.seco.i == l as u32 {
					reco.seco.i = b as u32;
				}
				reco.third = update_third_nearest_without_new(mat, med, reco.near.i as usize, reco.seco.i as usize, b, o);
			} else {
				if reco.near.i == l as u32 {
					reco.near.i = b as u32;
				}
				if reco.seco.i == l as u32 {
					reco.seco.i = b as u32;
				}
				if reco.third.i == l as u32 {
					reco.third.i = b as u32;
				}
			}
			_loss::<N, L>(reco.near.d.clone(), reco.seco.d.clone())
		})
		.reduce(L::add)
		.unwrap()
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{dynmsc, silhouette, medoid_silhouette, initialization::random_initialization};

	#[test]
	fn testdynmsc_simple() {
		let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
		let mut meds = random_initialization(4, 3, &mut rand::thread_rng());
		let (loss, assi, n_iter, n_swap, best_meds, losses): (f64, _, _, _, _, _) = dynmsc(&data, &mut meds, 2,100);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		let (msil, _): (f64, _) = medoid_silhouette(&data, &best_meds, false);
		print!("DynMSC_simple: {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, msil, sil, assi, best_meds, losses);
		assert_eq!(loss, 0.9375, "loss not as expected");
		assert_eq!(msil, 0.9375, "Medoid Silhouette not as expected");
		assert_eq!(best_meds.len(), 3, "Best k not as expected");
	}
	#[test]
	fn testdynmsc_mink() {
		let data = ndarray::arr2(&[[0,1,2,3,1],[1,0,4,5,2],[2,4,0,6,3],[3,5,6,0,4],[2,1,5,6,5]]);
		let mut meds = random_initialization(5, 3, &mut rand::thread_rng());
		let (loss, assi, n_iter, n_swap, best_meds, losses): (f64, _, _, _, _, _) = dynmsc(&data, &mut meds, 1,100);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		let (msil, _): (f64, _) = medoid_silhouette(&data, &best_meds, false);
		print!("DynMSC_mink: {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, msil, sil, assi, best_meds, losses);
		assert_eq!(loss, 0.87, "loss not as expected");
		assert_eq!(msil, 0.87, "Medoid Silhouette not as expected");
		assert_eq!(best_meds.len(), 3, "Best k not as expected");
	}
}
