use crate::arrayadapter::ArrayAdapter;
use crate::util::*;
use core::ops::AddAssign;
use num_traits::{Signed, Zero, Float};
use std::convert::From;

#[inline]
fn _loss<N, L>(a: N, b: N) -> L
	where
		N: Zero + Clone,
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
	med: &mut [usize],
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
	where
		N: Zero + PartialOrd + Clone,
		L: Float + Signed + AddAssign + From<N> + From<u32>,
		M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	if k == 2 { // special hadling, as there is no third
		return fastermsc_k2(mat, med, maxiter);
	}
	let (mut loss, mut data): (L, _) = initial_assignment(mat, med);
	debug_assert_assignment_th(mat, med, &data);

	let mut removal_loss = vec![L::zero(); k];
	update_removal_loss(&data, &mut removal_loss);
	let (mut lastswap, mut n_swaps, mut iter) = (n, 0, 0);
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
			loss = do_swap(mat, med, &mut data, b, j);
			update_removal_loss(&data, &mut removal_loss);
		}
		if n_swaps == swaps_before || loss >= lastloss {
			break; // converged
		}
	}
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	loss = L::one() - loss / <L as From<u32>>::from(n as u32);
	(loss, assi, iter, n_swaps)
}

/// Perform the initial assignment to medoids
#[inline]
pub(crate) fn initial_assignment<M, N, L>(mat: &M, med: &[usize]) -> (L, Vec<Reco<N>>)
	where
		N: Zero + PartialOrd + Clone,
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
					cur.third = cur.seco.clone();
					cur.seco = cur.near.clone();
					cur.near = DistancePair { i: m as u32, d };
				} else if cur.seco.i == u32::MAX || d < cur.seco.d {
					cur.third = cur.seco.clone();
					cur.seco = DistancePair { i: m as u32, d };
				} else if cur.third.i == u32::MAX || d < cur.third.d {
					cur.third = DistancePair { i: m as u32, d };
				}
			}
			_loss::<N, L>(cur.near.d.clone(), cur.seco.d.clone())
		})
		.reduce(L::add)
		.unwrap();
	(loss, data)
}

/// Find the best swap for object j - FastMSC version
#[inline]
pub(crate) fn find_best_swap<M, N, L>(
	mat: &M,
	removal_loss: &[L],
	data: &[Reco<N>],
	j: usize,
) -> (L, usize)
	where
		N: Zero + PartialOrd + Clone,
		L: Float + AddAssign + From<N>,
		M: ArrayAdapter<N>,
{
	let mut ploss = removal_loss.to_vec();
	// Improvement from the journal version:
	let mut acc = L::zero();
	for (o, reco) in data.iter().enumerate() {
		let doj = mat.get(o, j);
		if doj < reco.near.d {
			acc += _loss::<N, L>(reco.near.d.clone(), reco.seco.d.clone()) - _loss::<N, L>(doj.clone(), reco.near.d.clone());
			// loss already includes (dt - ds) - (ds - dn), remove
			ploss[reco.near.i as usize] += _loss::<N, L>(doj.clone(), reco.near.d.clone()) + _loss::<N, L>(reco.seco.d.clone(), reco.third.d.clone()) - _loss::<N, L>(reco.near.d.clone() + doj.clone(), reco.seco.d.clone());
			ploss[reco.seco.i as usize] += _loss::<N, L>(reco.near.d.clone(), reco.third.d.clone()) - _loss::<N, L>(reco.near.d.clone(), reco.seco.d.clone());
		} else if doj < reco.seco.d {
			acc += _loss::<N, L>(reco.near.d.clone(), reco.seco.d.clone()) - _loss::<N, L>(reco.near.d.clone(), doj.clone());
			ploss[reco.near.i as usize] += _loss::<N, L>(reco.near.d.clone(), doj.clone()) + _loss::<N, L>(reco.seco.d.clone(), reco.third.d.clone()) - _loss::<N, L>(reco.near.d.clone() + doj.clone(), reco.seco.d.clone());
			// loss already includes (dt - ds) - (ds - dn), adjust to 2*d(xo) - ds - dt
			// loss already includes (dt - ds), adjust to 2*d(xo) - ds - dt
			ploss[reco.seco.i as usize] += _loss::<N, L>(reco.near.d.clone(), reco.third.d.clone()) - _loss::<N, L>(reco.near.d.clone(), reco.seco.d.clone());
		} else if doj < reco.third.d {
			// loss already includes (dt - ds) - (ds - dn), adjust to d(xo)- dt
			ploss[reco.near.i as usize] += _loss::<N, L>(reco.seco.d.clone(), reco.third.d.clone()) - _loss::<N, L>(reco.seco.d.clone(), doj.clone());
			// loss already includes (dt - ds), adjust to d(xo)- dt
			ploss[reco.seco.i as usize] += _loss::<N, L>(reco.near.d.clone(), reco.third.d.clone()) - _loss::<N, L>(reco.near.d.clone(), doj.clone());
		}
	}
	let (b, bloss) = find_max(&mut ploss.iter());
	(bloss + acc, b) // add the shared accumulator
}

/// Update the loss when removing each medoid
pub(crate) fn update_removal_loss<N, L>(data: &[Reco<N>], loss: &mut [L])
	where
		N: Zero + Clone,
		L: Float + Signed + AddAssign + From<N>,
{
	loss.fill(L::zero()); // stable since 1.50
	for rec in data.iter() {
		loss[rec.near.i as usize] += _loss::<N, L>(rec.near.d.clone(), rec.seco.d.clone()) - _loss::<N, L>(rec.seco.d.clone(), rec.third.d.clone());
		loss[rec.seco.i as usize] += _loss::<N, L>(rec.near.d.clone(), rec.seco.d.clone()) - _loss::<N, L>(rec.near.d.clone(), rec.third.d.clone());
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
	doj: N,
) -> DistancePair<N>
	where
		N: PartialOrd + Clone,
		M: ArrayAdapter<N>,
{
	let mut dist = DistancePair::new(b as u32, doj);
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
	med: &mut [usize],
	data: &mut [Reco<N>],
	b: usize,
	j: usize,
) -> L
	where
		N: Zero + PartialOrd + Clone,
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
						reco.third = reco.seco.clone();
					}
					reco.seco = reco.near.clone();
				}
				reco.near = DistancePair::new(b as u32, N::zero());
				return L::zero();
			}
			let doj = mat.get(o, j);
			// Nearest medoid is gone:
			if reco.near.i == b as u32 {
				if doj < reco.seco.d {
					reco.near = DistancePair::new(b as u32, doj);
				} else if reco.third.i == u32::MAX || doj < reco.third.d {
					reco.near = reco.seco.clone();
					reco.seco = DistancePair::new(b as u32, doj);
				} else {
					reco.near = reco.seco.clone();
					reco.seco = reco.third.clone();
					reco.third = update_third_nearest(mat, med, reco.near.i as usize, reco.seco.i as usize, b, o, doj);
				}
			} else if reco.seco.i == b as u32 {
				// second nearest was replaced
				if doj < reco.near.d {
					reco.seco = reco.near.clone();
					reco.near = DistancePair::new(b as u32, doj);
				} else if reco.third.i == u32::MAX || doj < reco.third.d {
					reco.seco = DistancePair::new(b as u32, doj);
				} else {
					reco.seco = reco.third.clone();
					reco.third = update_third_nearest(mat, med, reco.near.i as usize, reco.seco.i as usize, b, o, doj);
				}
			} else {
				// nearest not removed
				if doj < reco.near.d {
					reco.third = reco.seco.clone();
					reco.seco = reco.near.clone();
					reco.near = DistancePair::new(b as u32, doj);
				} else if doj < reco.seco.d {
					reco.third = reco.seco.clone();
					reco.seco = DistancePair::new(b as u32, doj);
				} else if reco.third.i == u32::MAX || doj < reco.third.d {
					reco.third = DistancePair::new(b as u32, doj);
				} else if reco.third.i == b as u32 {
					reco.third = update_third_nearest(mat, med, reco.near.i as usize, reco.seco.i as usize, b, o, doj);
				}
			}
			_loss::<N, L>(reco.near.d.clone(), reco.seco.d.clone())
		})
		.reduce(L::add)
		.unwrap()
}

/// Special case k=2 of the FasterMSC algorithm.
pub(crate) fn fastermsc_k2<M, N, L>(
	mat: &M,
	med: &mut [usize],
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
	where
		N: Zero + PartialOrd + Clone,
		L: Float + Signed + AddAssign + From<N> + From<u32>,
		M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	assert!(k == 2, "Only valid for k=2");
	let (mut loss, mut assi, mut data): (L,_,_) = initial_assignment_k2(mat, med);
	let (mut lastswap, mut n_swaps, mut iter) = (n, 0, 0);
	while iter < maxiter {
		iter += 1;
		let (swaps_before, lastloss) = (n_swaps, loss);
		for j in 0..n {
			if j == lastswap {
				break;
			}
			if j == med[assi[j]] {
				continue; // This already is a medoid
			}
			let (newloss, b): (L, _) = find_best_swap_k2(mat, &data, j); // assi not used, see below
			if newloss >= loss {
				continue; // No improvement
			}
			n_swaps += 1;
			lastswap = j;
			// perform the swap
			loss = do_swap_k2(mat, med, &mut assi, &mut data, b, j);
		}
		if n_swaps == swaps_before || loss >= lastloss {
			break; // converged
		}
	}
	loss = L::one() - loss / <L as From<u32>>::from(n as u32);
	(loss, assi, iter, n_swaps)
}

/// Perform the initial assignment to medoids, for k=2 only
#[inline]
pub(crate) fn initial_assignment_k2<M, N, L>(mat: &M, med: &[usize]) -> (L, Vec<usize>, Vec<(N,N)>)
	where
		N: Zero + PartialOrd + Clone,
		L: Float + AddAssign + From<N>,
		M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), med.len());
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k == 2, "k must be 2");
	let mut assi = vec![0_usize; mat.len()];
	let mut data = vec![(N::zero(), N::zero()); mat.len()];
	let loss = assi.iter_mut().zip(data.iter_mut())
		.enumerate()
		.map(|(i, (a, d))| {
			*d = (mat.get(i, med[0]), mat.get(i, med[1]));
			if d.0 < d.1 {
				*a = 0;
				_loss::<N, L>(d.0.clone(), d.1.clone()) // return
			} else {
				*a = 1;
				_loss::<N, L>(d.1.clone(), d.0.clone()) // return
			}
		})
		.reduce(L::add)
		.unwrap();
	(loss, assi, data)
}

/// Find the best swap for object j - FastMSC version
#[inline]
pub(crate) fn find_best_swap_k2<M, N, L>(
	mat: &M,
	data: &[(N, N)],
	j: usize,
) -> (L, usize)
	where
		N: Zero + PartialOrd + Clone,
		L: Float + AddAssign + From<N>,
		M: ArrayAdapter<N>,
{
	let mut ploss = [L::zero(); 2];
	for (o, d) in data.iter().enumerate() {
		let doj = mat.get(o, j);
		// We do not use the assignment here, because we stored d0/d1 by medoid position, not closeness
		ploss[0] += if doj < d.1 { _loss(doj.clone(), d.1.clone()) } else { _loss(d.1.clone(), doj.clone()) };
		ploss[1] += if doj < d.0 { _loss(doj.clone(), d.0.clone()) } else { _loss(d.0.clone(), doj.clone()) };
	}
	let (b, bloss) = find_min(&mut ploss.iter());
	(bloss, b)
}

/// Perform a single swap
#[inline]
pub(crate) fn do_swap_k2<M, N, L>(
	mat: &M,
	med: &mut [usize],
	assi: &mut [usize],
	data: &mut [(N, N)],
	b: usize,
	j: usize,
) -> L
	where
		N: Zero + PartialOrd + Clone,
		L: Float + Signed + AddAssign + From<N>,
		M: ArrayAdapter<N>,
{
	let n = mat.len();
	assert!(b < med.len(), "invalid medoid number");
	assert!(j < n, "invalid object number");
	med[b] = j;
	// Its nicer to have the if outside, even though this looks duplicated
	if b == 0 {
		return assi.iter_mut().zip(data.iter_mut())
			.enumerate()
			.map(|(o, (a, d))| {
				if o == j {
					*a = 0;
					d.0 = N::zero();
					return L::zero();
				}
				let doj = mat.get(o, j);
				d.0 = doj.clone();
				if doj < d.1 || (doj == d.1 && *a == 0) {
					*a = 0;
					_loss::<N, L>(doj, d.1.clone()) // return
				} else {
					*a = 1;
					_loss::<N, L>(d.1.clone(), doj) // return
				}
			})
			.reduce(L::add)
			.unwrap();
	} else { // b == 1
		return assi.iter_mut().zip(data.iter_mut())
			.enumerate()
			.map(|(o, (a, d))| {
				if o == j {
					*a = 1;
					d.1 = N::zero();
					return L::zero();
				}
				let doj = mat.get(o, j);
				d.1 = doj.clone();
				if doj < d.0 || (doj == d.0 && *a == 1) {
					*a = 1;
					_loss::<N, L>(doj, d.0.clone()) // return
				} else {
					*a = 0;
					_loss::<N, L>(d.0.clone(), doj) // return
				}
			})
			.reduce(L::add)
			.unwrap();
	}
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{arrayadapter::LowerTriangle, fastermsc, silhouette, medoid_silhouette, util::assert_array};

	#[test]
	fn testfastermsc_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1, 2];
		let (loss, assi, n_iter, n_swap): (f64, _, _, _) = fastermsc(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		let (msil, _): (f64, _) = medoid_silhouette(&data, &meds, false);
		print!("FasterMSC: {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, sil, assi, meds);
		assert_eq!(loss, 0.9047619047619048, "loss not as expected");
		assert_eq!(msil, 0.9047619047619048, "Medoid Silhouette not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 2, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3, 2], "medoids not as expected");
		assert_eq!(sil, 0.5622222222222222, "Silhouette not as expected");
	}

	#[test]
	fn testfastermsc_simple2() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap): (f64, _, _, _) = fastermsc(&data, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		let (msil, _): (f64, _) = medoid_silhouette(&data, &meds, false);
		print!("FasterMSC: {:?} {:?} {:?} {:?} {:?} {:?}", loss, n_iter, n_swap, sil, assi, meds);
		assert_eq!(loss, 0.8805555555555555, "loss not as expected");
		assert_eq!(msil, 0.8805555555555555, "Medoid Silhouette not as expected");
		assert_eq!(n_swap, 3, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 4], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}
}
