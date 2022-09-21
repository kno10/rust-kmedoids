use crate::arrayadapter::ArrayAdapter;
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

/// Run the FasterMSC algorithm.
///
/// If used multiple times, it is better to additionally shuffle the input data,
/// to increase randomness of the solutions found and hence increase the chance
/// of finding a better solution.
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
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::fastermsc(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn fastermsc<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
	where
		N: Zero + PartialOrd + Copy + std::fmt::Display,
		L: Float + Signed + AddAssign + From<N> + From<u32> + std::fmt::Display,
		M: ArrayAdapter<N>,
{
	let n = mat.len();
	let k = med.len();
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let (mut loss, mut data):(L,_) = initial_assignment(mat, med);
	debug_assert_assignment_th(mat, med, &data);

	let mut removal_loss = vec![L::zero(); k];
	update_removal_loss(&data, &mut removal_loss);
	let (mut lastswap, mut n_swaps, mut iter) = (n, 0, 0);
	while iter < maxiter {
		iter += 1;
		let swaps_before = n_swaps;
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
			let newloss = do_swap(mat, med, &mut data, b, j);
			if newloss <= loss {
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
	loss = loss / <L as From<u32>>::from(n as u32);
	(loss, assi, iter, n_swaps)
}

/// Perform the initial assignment to medoids
#[inline]
pub(crate) fn initial_assignment<M, N, L>(mat: &M, med: &[usize]) -> (L, Vec<Reco<N>>)
	where
		N: Zero + PartialOrd + Copy,
		L: Float + AddAssign + From<N>,
		M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	let mut data = vec![Reco::<N>::empty(); mat.len()];
	let firstcenter = med[0];
	let loss = data
		.iter_mut()
		.enumerate()
		.map(|(i, cur)| {
			*cur = Reco::new(0, mat.get(i, firstcenter), u32::MAX, N::zero(), u32::MAX, N::zero());
			for (m, &me) in med.iter().enumerate().skip(1) {
				let d = mat.get(i, me);
				if d < cur.near.d || i == me {
					cur.third = cur.seco;
					cur.seco = cur.near;
					cur.near = DistancePair { i: m as u32, d };
				} else if cur.seco.i == u32::MAX || d < cur.seco.d {
					cur.third = cur.seco;
					cur.seco = DistancePair { i: m as u32, d };
				} else if cur.third.i == u32::MAX || d < cur.third.d {
					cur.third = DistancePair { i: m as u32, d };
				}
			}
			L::one() - _loss::<N, L>(cur.near.d, cur.seco.d)
		})
		.reduce(L::add)
		.unwrap();
	(loss, data)
}

/// Find the best swap for object j - FasterMSC version
#[inline]
pub(crate) fn find_best_swap<M, N, L>(
	mat: &M,
	removal_loss: &[L],
	data: &[Reco<N>],
	j: usize,
) -> (L, usize)
	where
		N: Zero + PartialOrd + Copy + std::fmt::Display,
		L: Float + AddAssign + From<N> + std::fmt::Display,
		M: ArrayAdapter<N>,
{
	let mut ploss = removal_loss.to_vec();
	// Improvement from the journal version:
	let mut acc = L::zero();
	for (o, reco) in data.iter().enumerate() {
		let djo = mat.get(j, o);
		if djo < reco.near.d {
			acc += _loss::<N, L>(reco.near.d, reco.seco.d) - _loss::<N, L>(djo, reco.near.d);
			// loss already includes (dt - ds) - (ds - dn), remove
			ploss[reco.near.i as usize] += _loss::<N, L>(djo, reco.near.d) + _loss::<N, L>(reco.seco.d, reco.third.d) - _loss::<N, L>(reco.near.d + djo, reco.seco.d);
			ploss[reco.seco.i as usize] += _loss::<N, L>(reco.near.d, reco.third.d) - _loss::<N, L>(reco.near.d, reco.seco.d);
		} else if djo < reco.seco.d {
			acc += _loss::<N, L>(reco.near.d, reco.seco.d) - _loss::<N, L>(reco.near.d, djo);
			ploss[reco.near.i as usize] += _loss::<N, L>(reco.near.d, djo) + _loss::<N, L>(reco.seco.d, reco.third.d) - _loss::<N, L>(reco.near.d + djo, reco.seco.d);
			ploss[reco.seco.i as usize] += _loss::<N, L>(reco.near.d, reco.third.d) - _loss::<N, L>(reco.near.d, reco.seco.d);
		} else if djo < reco.third.d {
			ploss[reco.near.i as usize] += _loss::<N, L>(reco.seco.d, reco.third.d) - _loss::<N, L>(reco.seco.d, djo);
			ploss[reco.seco.i as usize] += _loss::<N, L>(reco.near.d, reco.third.d) - _loss::<N, L>(reco.near.d, djo);
		}
	}
	let (b, bloss) = find_max(&mut ploss.iter());
	(bloss + acc, b) // add the shared accumulator
}

/// Update the loss when removing each medoid
pub(crate) fn update_removal_loss<N, L>(data: &[Reco<N>], loss: &mut Vec<L>)
	where
		N: Zero + Copy,
		L: Float + Signed + AddAssign + From<N>,
{
	loss.fill(L::zero()); // stable since 1.50
	for rec in data.iter() {
		loss[rec.near.i as usize] += _loss::<N, L>(rec.near.d, rec.seco.d) - _loss::<N, L>(rec.seco.d, rec.third.d);
		loss[rec.seco.i as usize] += _loss::<N, L>(rec.near.d, rec.seco.d) - _loss::<N, L>(rec.near.d, rec.third.d);
		// as N might be unsigned
	}
}

/// Update the third nearest medoid information
/// Called after each swap.
#[inline]
pub(crate) fn update_third_nearest<M, N>(
	mat: &M,
	med: &[usize],
	n: usize,
	s: usize,
	b: usize,
	o: usize,
	djo: N,
) -> DistancePair<N>
	where
		N: PartialOrd + Copy,
		M: ArrayAdapter<N>,
{
	let mut dist = DistancePair::new(b as u32, djo);
	for (i, &mi) in med.iter().enumerate() {
		if i == n || i == b || i == s {
			continue;
		}
		let d = mat.get(o, mi);
		if d < dist.d {
			dist = DistancePair::new(i as u32, d);
		}
	}
	dist
}

/// Perform a single swap
#[inline]
pub(crate) fn do_swap<M, N, L>(
	mat: &M,
	med: &mut Vec<usize>,
	data: &mut Vec<Reco<N>>,
	b: usize,
	j: usize,
) -> L
	where
		N: Zero + PartialOrd + Copy,
		L: Float + Signed + AddAssign + From<N>,
		M: ArrayAdapter<N>,
{
	let n = mat.len();
	assert!(b < med.len(), "invalid medoid number");
	assert!(j < n, "invalid object number");
	med[b] = j;
	data.iter_mut()
		.enumerate()
		.map(|(o, reco)| {
			if o == j {
				if reco.near.i != b as u32 {
					if reco.seco.i != b as u32 {
						reco.third = reco.seco;
					}
					reco.seco = reco.near;
				}
				reco.near = DistancePair::new(b as u32, N::zero());
				return L::one();
			}
			let djo = mat.get(j, o);
			// Nearest medoid is gone:
			if reco.near.i == b as u32 {
				if djo < reco.seco.d {
					reco.near = DistancePair::new(b as u32, djo);
				} else if djo < reco.third.d{
					reco.near = reco.seco;
					reco.seco = DistancePair::new(b as u32, djo);
				} else {
					reco.near = reco.seco;
					reco.seco = reco.third;
					reco.third = update_third_nearest(mat, med, reco.near.i as usize, reco.seco.i as usize, b, o, djo);
				}
			} else if reco.seco.i == b as u32{
				// second nearest was replaced
				if djo < reco.near.d {
					reco.seco = reco.near;
					reco.near = DistancePair::new(b as u32, djo);
				} else if djo < reco.third.d {
					reco.seco = DistancePair::new(b as u32, djo);
				} else {
					reco.seco = reco.third;
					reco.third = update_third_nearest(mat, med, reco.near.i as usize, reco.seco.i as usize, b, o, djo);
				}
			} else {
				// nearest not removed
				if djo < reco.near.d {
					reco.third = reco.seco;
					reco.seco = reco.near;
					reco.near = DistancePair::new(b as u32, djo);
				} else if djo < reco.seco.d {
					reco.third = reco.seco;
					reco.seco = DistancePair::new(b as u32, djo);
				} else if djo < reco.third.d {
					reco.third = DistancePair::new(b as u32, djo);
				} else if reco.third.i == b as u32 {
					reco.third = update_third_nearest(mat, med, reco.near.i as usize, reco.seco.i as usize, b, o, djo);
				}
			}
			L::one() - _loss::<N, L>(reco.near.d, reco.seco.d)
		})
		.reduce(L::add)
		.unwrap()
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{arrayadapter::LowerTriangle, fastermsc, silhouette, util::assert_array};

	#[test]
	fn testfastermsc_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1, 2];
		let (loss, assi, n_iter, n_swap): (f64, _, _, _) = fastermsc(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		print!("Faster: {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, sil, assi, meds);
		assert_eq!(loss, 0.9047619047619048, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 2, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3, 2], "medoids not as expected");
		assert_eq!(sil, 0.8773115773115773, "Silhouette not as expected");
	}
}
