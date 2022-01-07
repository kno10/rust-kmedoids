use crate::arrayadapter::ArrayAdapter;
use crate::silhouette::checked_div;
use core::ops::{AddAssign, Div, Sub};
use num_traits::{Signed, Zero};
use rayon::prelude::*;
use std::convert::From;

/// Compute the Silhouette of a strict partitional clustering (parallel implementation).
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
/// let sil: f64 = kmedoids::par_silhouette(&data, &assi);
/// println!("Silhouette is: {}", sil);
/// ```
#[cfg(feature = "parallel")]
pub fn par_silhouette<M, N, L>(mat: &M, assi: &[usize]) -> L
where
	N: Zero + PartialOrd + Copy + Sync + Send,
	L: AddAssign
		+ Div<Output = L>
		+ Sub<Output = L>
		+ Signed
		+ Zero
		+ PartialOrd
		+ Copy
		+ From<N>
		+ From<u32>
		+ Sync
		+ Send,
	M: ArrayAdapter<N> + Sync + Send,
{
	let mut lsum = L::zero();
	assi.into_par_iter()
		.enumerate()
		.map(|(i, &ai)| {
			let mut buf = Vec::<(u32, L)>::new();
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
			let a = checked_div(buf[ai].1, buf[ai].0.into());
			let mut tmp = buf
				.iter()
				.enumerate()
				.filter(|&(i, _)| i != ai)
				.map(|(_, p)| checked_div(p.1, p.0.into()));
			// Ugly hack to get the min():
			let tmp2 = tmp.next().unwrap_or_else(L::zero);
			let b = tmp.fold(tmp2, |x, y| if y < x { x } else { y });
			checked_div(b - a, if a > b { a } else { b }) // return value
		})
		.collect::<Vec<L>>()
		.iter()
		.for_each(|x| lsum += *x);
	lsum.div((assi.len() as u32).into())
}
