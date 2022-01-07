use crate::arrayadapter::ArrayAdapter;
use crate::util::*;
use core::ops::AddAssign;
use num_traits::{Signed, Zero};
use std::convert::From;

/// Run the Alternating algorithm, a k-means-style alternate optimization.
///
/// This is fairly fast (O(n²), like the FasterPAM method), but because the
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
pub fn alternating<M, N, L>(mat: &M, med: &mut [usize], maxiter: usize) -> (L, Vec<usize>, usize)
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let mut assi = vec![usize::MAX; mat.len()];
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

/// Assign each to the nearest medoid, return loss
#[inline]
#[allow(clippy::or_fun_call)] // zero() is fine
fn assign_nearest<M, N, L>(mat: &M, med: &[usize], data: &mut [usize]) -> L
where
	N: PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	debug_assert!(data.len() == n, "data not preallocated");
	let firstcenter = med[0];
	data.iter_mut()
		.enumerate()
		.map(|(i, di)| {
			let mut best = (0, mat.get(i, firstcenter));
			for (m, &mm) in med.iter().enumerate().skip(1) {
				let dm = mat.get(i, mm);
				if dm < best.1 || i == mm {
					best = (m, dm);
				}
			}
			*di = best.0;
			L::from(best.1)
		})
		.reduce(L::add)
		.unwrap_or(L::zero())
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{alternating, arrayadapter::LowerTriangle, silhouette, util::assert_array};

	#[test]
	fn test_alternating() {
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
