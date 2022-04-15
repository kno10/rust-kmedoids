use crate::arrayadapter::ArrayAdapter;
use crate::fasterpam::{do_swap, initial_assignment};
use crate::util::*;
use core::ops::AddAssign;
use num_traits::{Signed, Zero};
use std::convert::From;

/// Run the original PAM SWAP algorithm (no BUILD, but given initial medoids).
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
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
	M: ArrayAdapter<N>,
{
	let (loss, mut data) = initial_assignment(mat, med);
	pam_optimize(mat, med, &mut data, maxiter, loss)
}

/// Run the original PAM BUILD algorithm.
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
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
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

/// Run the original PAM algorithm (BUILD and SWAP).
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
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
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

/// Main optimization function of PAM, not exposed (use pam_swap or pam)
fn pam_optimize<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	data: &mut Vec<Rec<N>>,
	maxiter: usize,
	mut loss: L,
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
	debug_assert_assignment(mat, med, data);
	let (mut n_swaps, mut iter) = (0, 0);
	while iter < maxiter {
		iter += 1;
		let mut best = (L::zero(), k, usize::MAX);
		for j in 0..n {
			if j == med[data[j].near.i as usize] {
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

/// Find the best swap for object j - slower PAM version
#[inline]
fn find_best_swap_pam<M, N, L>(mat: &M, med: &[usize], data: &[Rec<N>], j: usize) -> (L, usize)
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
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

/// Not exposed. Use pam_build or pam.
fn pam_build_initialize<M, N, L>(
	mat: &M,
	meds: &mut Vec<usize>,
	data: &mut Vec<Rec<N>>,
	k: usize,
) -> L
where
	N: Zero + PartialOrd + Copy,
	L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N>,
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
		data.push(Rec::new(0, mat.get(best.1, j), u32::MAX, N::zero()));
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
				recj.near = DistancePair::new(l as u32, N::zero());
				continue;
			}
			let dj = mat.get(best.1, j);
			if dj < recj.near.d {
				recj.seco = recj.near;
				recj.near = DistancePair::new(l as u32, dj);
			} else if recj.seco.i == u32::MAX || dj < recj.seco.d {
				recj.seco = DistancePair::new(l as u32, dj);
			}
			loss += L::from(recj.near.d);
		}
		meds.push(best.1);
	}
	loss
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{
		arrayadapter::LowerTriangle, pam, pam_build, pam_swap, silhouette, util::assert_array,
	};

	#[test]
	fn test_pam_swap_simple() {
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
	fn test_pam_build_simple() {
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
	fn test_pam_simple() {
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
}
