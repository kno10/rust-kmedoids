use crate::arrayadapter::ArrayAdapter;
use crate::fastermsc::{initial_assignment,update_removal_loss,find_best_swap,do_swap};
use crate::fastermsc::{initial_assignment_k2,find_best_swap_k2,do_swap_k2};
use crate::util::*;
use core::ops::AddAssign;
use num_traits::{Signed, Zero, Float};
use std::convert::From;

#[inline]
fn _loss<N, L>(a: N, b: N) -> L
	where
		N: Zero + Copy,
		L: Float + From<N>,
{
	if N::is_zero(&a) || N::is_zero(&b) { L::zero() } else { <L as From<N>>::from(a) / <L as From<N>>::from(b) } 
}

/// Run the FastMSC algorithm, which yields the same results as the original PAMMEDSIL.
///
/// This is faster than PAMMEDSIL, but slower than FasterMSC, and mostly of interest for academic reasons.
/// Quality-wise, FasterMSC is not worse on average, but much faster.
///
/// This is the improved version,
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
pub fn fastmsc<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
	where
		N: Zero + PartialOrd + Copy,
		L: Float + Signed + AddAssign + From<N> + From<u32>,
		M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	if k == 2 {
		return fastmsc_k2(mat, med, maxiter)
	}
	let (mut loss, mut data): (L,_) = initial_assignment(mat, med);
	debug_assert_assignment_th(mat, med, &data);

	let mut removal_loss = vec![L::zero(); k];
	let (mut n_swaps, mut iter) = (0, 0);
	while iter < maxiter {
		iter += 1;
		let mut best = (L::zero(), usize::MAX, usize::MAX);
		update_removal_loss(&data, &mut removal_loss);
		for j in 0..n {
			if j == med[data[j].near.i as usize] {
				continue; // This already is a medoid
			}
			let (change, b) = find_best_swap(mat, &removal_loss, &data, j);
			if change <= best.0 {
				continue; // No improvement
			}
			best = (change, b, j);
		}
		if best.0 > L::zero() {
			n_swaps += 1;
			// perform the swap
			let newloss = do_swap(mat, med, &mut data, best.1, best.2);
			if newloss <= loss {
				break; // Probably numerically unstable now.
			}
			loss = newloss;
		} else {
			break; // No improvement, or NaN.
		}
	}
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	loss = loss / <L as From<u32>>::from(n as u32);
	(loss, assi, iter, n_swaps)
}
/// Special case k=2 of the FasterMSC algorithm.
fn fastmsc_k2<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
	where
		N: Zero + PartialOrd + Copy,
		L: Float + Signed + AddAssign + From<N> + From<u32>,
		M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	assert!(k == 2, "Only valid for k=2");
	let (mut loss, mut assi, mut data): (L,_,_) = initial_assignment_k2(mat, med);
	let (mut n_swaps, mut iter) = (0, 0);
	while iter < maxiter {
		iter += 1;
		let mut best = (L::zero(), k, usize::MAX);
		for j in 0..n {
			if j == med[assi[j] as usize] {
				continue; // This already is a medoid
			}
			let (newloss, b): (L, _) = find_best_swap_k2(mat, &data, j); // assi not used, see below
			if newloss > best.0 {
				best = (newloss, b, j);
			}
		}
		if !(best.0 > loss) {
			break; // No improvement
		}
		// perform the swap
		n_swaps += 1;
		let newloss = do_swap_k2(mat, med, &mut assi, &mut data, best.1, best.2);
		if newloss <= loss {
			break; // Probably numerically unstable now.
		}
		loss = newloss;
	}
	loss = loss / <L as From<u32>>::from(n as u32);
	(loss, assi, iter, n_swaps)
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{arrayadapter::LowerTriangle, fastmsc, silhouette, medoid_silhouette, util::assert_array};

	#[test]
	fn testfastpammedsil_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1, 2];
		let (loss, assi, n_iter, n_swap): (f64, _, _, _) = fastmsc(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		let (msil, _): (f64, _) = medoid_silhouette(&data, &meds, false);
		print!("FastMSC: {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, sil, assi, meds);
		assert_eq!(loss, 0.9047619047619048, "loss not as expected");
		assert_eq!(msil, 0.9047619047619048, "Medoid Silhouette not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 2, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3, 2], "medoids not as expected");
		assert_eq!(sil, 0.8773115773115773, "Silhouette not as expected");
	}

	#[test]
	fn testfastpammedsil_simple2() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap): (f64, _, _, _) = fastmsc(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		let (msil, _): (f64, _) = medoid_silhouette(&data, &meds, false);
		print!("FastMSC: {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, sil, assi, meds);
		assert_eq!(loss, 0.8805555555555555, "loss not as expected");
		assert_eq!(msil, 0.8805555555555555, "Medoid Silhouette not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 4], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}
}
