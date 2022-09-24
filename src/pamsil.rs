use crate::arrayadapter::ArrayAdapter;
use crate::alternating::assign_nearest;
use crate::util::*;
use crate::silhouette::*;
use core::ops::AddAssign;
use num_traits::{Signed, Zero, Float};
use std::convert::From;

/// Run the original PAMSIL SWAP algorithm (no BUILD, but given initial medoids).
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
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::pamsil_swap(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn pamsil_swap<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
	where
		N: Zero + PartialOrd + Copy,
		L: Float + Signed + AddAssign + From<N> + From<u32>,
		M: ArrayAdapter<N>,
{
	let n = mat.len();
	let mut assi = vec![0; n];
	assign_nearest::<M, N, L>(mat, &med, &mut assi);
	let (nloss, n_iter, n_swap) = pamsil_optimize(mat, med, &mut assi, maxiter);
	(nloss, assi, n_iter, n_swap)
}

/// Run the original PAM BUILD algorithm combined with the PAMSIL SWAP.
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
/// let (loss, assi, meds, n_iter, n_swap): (f64, _, _, _, _) = kmedoids::pamsil(&data, 2, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn pamsil<M, N, L>(mat: &M, k: usize, maxiter: usize) -> (L, Vec<usize>, Vec<usize>, usize, usize)
	where
		N: Zero + PartialOrd + Copy,
		L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + From<u32>,
		M: ArrayAdapter<N>,
{
	let n = mat.len();
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	let mut meds = Vec::<usize>::with_capacity(k);
	let mut assi = vec![0; n];
	pamsil_build_initialize::<M, N, L>(mat, &mut meds, &mut assi, k);
	let (nloss, n_iter, n_swap) = pamsil_optimize(mat, &mut meds, &mut assi, maxiter);
	(nloss, assi, meds, n_iter, n_swap) // also return medoids
}

/// Main optimization function of PAMSIL, not exposed (use pamsil_swap or pamsil)
fn pamsil_optimize<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	assi: &mut Vec<usize>,
	maxiter: usize,
) -> (L, usize, usize)
	where
		N: Zero + PartialOrd + Copy,
		L: AddAssign + Signed + Zero + PartialOrd + Copy + From<N> + From<u32>,
		M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	if k == 1 {
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, 1, if swapped { 1 } else { 0 });
	}
	let (mut n_swaps, mut iter) = (0, 0);
	let (mut sil, _): (L, _) = silhouette::<M, N, L>(&mat, &assi, false);
	while iter < maxiter {
		iter += 1;
		let mut best = (L::zero(), k, usize::MAX);
		for m in 0..k {
			let medm = med[m]; // preseve previous value
			for j in 0..n {
				if j == medm || j == med[assi[j]] {
					continue; // This already is a medoid
				}
				med[m] = j; // replace
				assign_nearest::<M, N, L>(mat, &med, assi);
				let (siltemp, _): (L, _) = silhouette::<M, N, L>(&mat, &assi, false);
				if siltemp <= best.0 {
					continue; // No improvement
				}
				best = (siltemp, m, j);
			}
			med[m] = medm; // restore
		}
		if best.0 <= sil {
			break; // no improvement
		}
		n_swaps += 1;
		med[best.1] = best.2;
		sil = best.0;
	}
	assign_nearest::<M, N, L>(mat, &med, assi);
	(sil, iter, n_swaps)
}

/// Not exposed. Use pamsil_build or pamsil.
fn pamsil_build_initialize<M, N, L>(
	mat: &M,
	meds: &mut Vec<usize>,
	assi: &mut Vec<usize>,
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
	let mut data = Vec::<N>::with_capacity(n);
	assi.fill(0);
	for j in 0..n {
		data.push(mat.get(best.1, j));
	}
	// choose remaining medoids
	for _ in 1..k {
		best = (L::zero(), k);
		for (i, di) in data.iter().enumerate() {
			let mut sum = -L::from(*di);
			for (j, dnear) in data.iter().enumerate() {
				if j != i {
					let d = mat.get(i, j);
					if d < *dnear {
						sum += L::from(d) - L::from(*dnear)
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
		for (j, dnear) in data.iter_mut().enumerate() {
			if j == best.1 {
				*dnear = N::zero();
				continue;
			}
			let dj = mat.get(best.1, j);
			if dj < *dnear {
				*dnear = dj;
			}
			loss += L::from(*dnear);
		}
		meds.push(best.1);
	}
	loss
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{
		arrayadapter::LowerTriangle, pamsil, pamsil_swap, silhouette, util::assert_array,
	};

	#[test]
	fn test_pamsil() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let (loss, assi, meds, n_iter, n_swap): (f64, _, _, _, _) = pamsil(&data, 2, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		print!("PAMSil: {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, sil, assi, meds);
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_eq!(loss, 0.7522494172494172, "loss not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![1, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn test_pamsil3() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let (loss, assi, meds, n_iter, n_swap): (f64, _, _, _, _) = pamsil(&data, 3, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		print!("PAMSil k=3: {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, sil, assi, meds);
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_eq!(loss, 0.8773115773115773, "loss not as expected");
		assert_array(assi, vec![0, 0, 2, 1, 1], "assignment not as expected");
		assert_array(meds, vec![1, 3, 2], "medoids not as expected");
		assert_eq!(sil, 0.8773115773115773, "Silhouette not as expected");
	}

	#[test]
	fn testpamsil_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1, 2];
		let (loss, assi, n_iter, n_swap): (f64, _, _, _) = pamsil_swap(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		print!("Fast: {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, sil, assi, meds);
		assert_eq!(loss, 0.8773115773115773, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![1, 1, 2, 0, 0], "assignment not as expected");
		assert_array(meds, vec![3, 1, 2], "medoids not as expected");
		assert_eq!(sil, 0.8773115773115773, "Silhouette not as expected");
	}
}
