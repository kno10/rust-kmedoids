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
/// currently always returns a f64 result, and accepts only input distances that can be
/// converted into f64.
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
	N: Zero + PartialOrd + Copy,
	L: AddAssign
		+ Div<Output = L>
		+ Sub<Output = L>
		+ Signed
		+ Zero
		+ PartialOrd
		+ Copy
		+ From<N>
		+ From<u32>,
	M: ArrayAdapter<N>,
{
	let mut sil = if samples {
		vec![L::zero(); assi.len()]
	} else {
		vec![L::zero(); 0]
	};
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
		let a = checked_div(buf[ai].1, buf[ai].0.into());
		let mut tmp = buf
			.iter()
			.enumerate()
			.filter(|&(i, _)| i != ai)
			.map(|(_, p)| checked_div(p.1, p.0.into()));
		// Ugly hack to get the min():
		let tmp2 = tmp.next().unwrap_or_else(L::zero);
		let b = tmp.fold(tmp2, |x, y| if y < x { x } else { y });
		let s = checked_div(b - a, if a > b { a } else { b });
		if samples {
			sil[i] = s;
		}
		lsum += s;
	}
	if samples {
		assert_eq!(sil.len(), assi.len(), "Length not as expected.");
	}
	(lsum.div((assi.len() as u32).into()), sil)
}

// helper function, returns 0 on division by 0
pub(crate) fn checked_div<L>(x: L, y: L) -> L
where
	L: Div<Output = L> + Zero + Copy + PartialOrd,
{
	if y > L::zero() {
		x.div(y)
	} else {
		L::zero()
	}
}
