use crate::arrayadapter::ArrayAdapter;
use core::ops::{AddAssign, Div, Sub};
use num_traits::{Signed, Zero};
use std::convert::From;

/// Compute the Silhouette of a strict partitional clustering.
///
/// The Silhouette, proposed by Peter Rousseeuw in 1987, is a popular internal
/// evaluation measure for clusterings. Although it is defined on arbitary metrics,
/// it is most appropriate for evaluating "spherical" clusters, as it expects objects
/// to be closer to all members of its own cluster than to members of other clusters.
///
/// Because of the additional requirement of a division operator, this implementation
/// currently always returns a float result, and accepts only input distances that can be
/// converted into floats.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `f64` for the cost (use a float type)
/// * `mat` - a pairwise distance matrix
/// * `assi` - the cluster assignment
/// * `samples` - whether to keep the individual samples, or not
///
/// returns a tuple containing:
/// * the average silhouette
/// * the individual silhouette values (empty if `samples = false`)
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter): (f64, _, _) = kmedoids::alternating(&data, &mut meds, 100);
/// let (sil, _): (f64, _) = kmedoids::silhouette(&data, &assi, false);
/// println!("Silhouette is: {}", sil);
/// ```
pub fn silhouette<M, N, L>(mat: &M, assi: &[usize], samples: bool) -> (L, Vec<L>)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign
		+ Div<Output = L>
		+ Sub<Output = L>
		+ Signed
		+ Zero
		+ PartialOrd
		+ Clone
		+ From<N>
		+ From<u32>,
	M: ArrayAdapter<N>,
{
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	let mut sil =  vec![L::zero(); if samples { assi.len() } else { 0}];
	let mut lsum: L = L::zero();
	let mut buf = Vec::<(u32, L)>::new();
	for (i, &ai) in assi.iter().enumerate() {
		buf.clear();
		for (j, &aj) in assi.iter().enumerate() {
			while aj >= buf.len() {
				buf.push((0, L::zero()));
			}
			if i != j {
				buf[aj].0 += 1;
				buf[aj].1 += mat.get(i, j).into();
			}
		}
		if buf.len() == 1 {
			return (L::zero(), sil);
		}
		let s = if buf[ai].0 > 0 {
			let a = checked_div(buf[ai].1.clone(), buf[ai].0.into());
			let mut tmp = buf
				.iter()
				.enumerate()
				.filter(|&(i, _)| i != ai)
				.map(|(_, p)| checked_div(p.1.clone(), p.0.into()));
			// Ugly hack to get the min():
			let tmp2 = tmp.next().unwrap_or_else(L::zero);
			let b = tmp.fold(tmp2, |x, y| if y < x { y } else { x });
			checked_div(b.clone() - a.clone(), if a > b { a } else { b })
		} else {
			L::zero() // singleton
		};
		if samples {
			sil[i] = s.clone();
		}
		lsum += s.clone();
	}
	if samples {
		assert_eq!(sil.len(), assi.len(), "Length not as expected.");
	}
	(lsum.div((assi.len() as u32).into()), sil)
}

/// Compute the Medoid Silhouette of a clustering.
///
/// The Medoid Silhouette is an approximation to the original Silhouette where the
/// distance to the cluster medoid is used instead of the average distance, hence reducing
/// the run time from O(N²) to O(Nk). Here we assume that every object is assigned the
/// nearest cluster, and hence only a distance matrix and a list of medoids is given.
///
/// Because of the additional requirement of a division operator, this implementation
/// currently always returns a float result, and accepts only input distances that can be
/// converted into floats.
///
/// TODO: allow using N x k distance matrixes, too.
///
/// * type `M` - matrix data type such as `ndarray::Array2` or `kmedoids::arrayadapter::LowerTriangle`
/// * type `N` - number data type such as `u32` or `f64`
/// * type `L` - number data type such as `f64` for the cost (use a float type)
/// * `mat` - a pairwise distance matrix
/// * `meds` - the medoid list
/// * `samples` - whether to keep the individual samples, or not
///
/// returns a tuple containing:
/// * the average medoid silhouette
/// * the individual medoid silhouette values (empty if `samples = false`)
///
/// ## Panics
///
/// * panics when the dissimilarity matrix is not square
///
/// ## Example
/// Given a dissimilarity matrix of size 4 x 4, use:
/// ```
/// let data = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
/// let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
/// let (loss, assi, n_iter): (f64, _, _) = kmedoids::alternating(&data, &mut meds, 100);
/// let (sil, _): (f64, _) = kmedoids::medoid_silhouette(&data, &meds, false);
/// println!("Silhouette is: {}", sil);
/// ```
pub fn medoid_silhouette<M, N, L>(mat: &M, meds: &[usize], samples: bool) -> (L, Vec<L>)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign
		+ Div<Output = L>
		+ Sub<Output = L>
		+ Signed
		+ Zero
		+ PartialOrd
		+ Clone
		+ From<N>
		+ From<u32>,
	M: ArrayAdapter<N>,
{
	let (n, k) = (mat.len(), meds.len());
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	let mut sil = vec![L::one(); if samples { n } else { 0 }];
	if k == 1 { return (L::one(), sil); } // not really well-defined
	assert!(k <= n, "invalid k, must be over 1 and at most N");
	let mut loss = L::zero();
	#[allow(clippy::needless_range_loop)]
	for i in 0..n {
		let (d1, d2) = (mat.get(i, meds[0]), mat.get(i, meds[1]));
		let mut best = if d1 < d2 { (d1, d2) } else { (d2, d1) };
		for &m in meds.iter().skip(2) {
			let d = mat.get(i, m);
			if d < best.0 {
				best = (d, best.0);
			}
			else if d < best.1 {
				best = (best.0, d);
			}
		}
		if !N::is_zero(&best.0) {
			let s = <L as From<N>>::from(best.0.clone()) / <L as From<N>>::from(best.1.clone());
			if samples { sil[i] = L::one() - s.clone(); }
			loss += s;
		}
	}
	loss = L::one() - loss / <L as From<u32>>::from(n as u32);
	(loss, sil)
}

// helper function, returns 0 on division by 0
pub(crate) fn checked_div<L>(x: L, y: L) -> L
where
	L: Div<Output = L> + Zero + Clone + PartialOrd,
{
	if y > L::zero() {
		x.div(y)
	} else {
		L::zero()
	}
}
