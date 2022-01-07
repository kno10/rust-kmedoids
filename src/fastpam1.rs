use crate::arrayadapter::ArrayAdapter;
use crate::fasterpam::{do_swap, find_best_swap, initial_assignment, update_removal_loss};
use crate::util::*;
use core::ops::AddAssign;
use num_traits::{Signed, Zero};
use std::convert::From;

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
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let (mut loss, mut data) = initial_assignment(mat, med);
	debug_assert_assignment(mat, med, &data);
	let mut removal_loss = vec![L::zero(); k];
	let (mut n_swaps, mut iter) = (0, 0);
	while iter < maxiter {
		iter += 1;
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
			if newloss >= loss {
				break; // Probably numerically unstable now.
			}
			loss = newloss;
		} else {
			break; // No improvement, or NaN.
		}
	}
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, iter, n_swaps)
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{arrayadapter::LowerTriangle, fastpam1, silhouette, util::assert_array};

	#[test]
	fn test_fastpam1_simple() {
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
}
