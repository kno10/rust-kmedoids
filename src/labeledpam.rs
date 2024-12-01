use crate::arrayadapter::ArrayAdapter;
use crate::arrayadapter::LabelAdapter;
use crate::util::*;
use core::ops::AddAssign;
use num_traits::{Signed, Zero};
use std::convert::From;

/// Run the LabeledPAM algorithm.
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
/// let (loss, assi, n_iter, n_swap): (f64, _, _, _) = kmedoids::fasterpam(&data, &mut meds, 100);
/// println!("Loss is: {}", loss);
/// ```
pub fn labeledpam<M, N, L,O>(
	mat: &M,
	labels: &O,
	med: &mut [usize],
	maxiter: usize,
) -> (L, Vec<usize>, usize, usize)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Signed + Zero + PartialOrd + Clone + From<N> + Copy,
	M: ArrayAdapter<N>,
	O: LabelAdapter<usize>,
{
	let (n, k, l) = (mat.len(), med.len(), labels.len());
	assert!(l<k, "Too many labels for the number of medoids");
	if k == 1 {
		let assi = vec![0; n];
		let (swapped, loss) = choose_medoid_within_partition::<M, N, L>(mat, &assi, med, 0);
		return (loss, assi, 1, if swapped { 1 } else { 0 });
	}
	let mut cluster_records = CluRec::<usize>::new(med, l);
	let (mut loss, mut data): (L, Vec<Rec<N>>) = initial_assignment(mat, labels, &mut cluster_records);
	debug_assert_assignment(mat, &cluster_records.meds, &data);
	let mut removal_loss = vec![L::zero(); k];
	update_removal_loss(&mut data, &mut removal_loss, mat, labels, &mut cluster_records);
	let (mut lastswap, mut n_swaps, mut iter) = (n, 0, 0);
	while iter < maxiter {
		iter += 1;
		let (swaps_before, lastloss) = (n_swaps, loss.clone());
		for j in 0..n {
			if j == lastswap {
				break;
			}
			if j == cluster_records.meds[data[j].near.i as usize] {
				continue; // This already is a medoid
			}
			let (b, l, change) = find_best_swap(mat, labels, &removal_loss, &data, &cluster_records, j);
			if change >= L::zero() {
				continue; // No improvement
			}
			n_swaps += 1;
			lastswap = j;
			// perform the swap
			loss = do_swap(mat, labels, &mut data, &mut cluster_records, j, b, l);
			update_removal_loss(&mut data, &mut removal_loss, mat, labels, &mut cluster_records);
		}
		if n_swaps == swaps_before || loss >= lastloss {
			break; // converged
		}
	}
	let assi = data.iter().map(|x| x.near.i as usize).collect();
	(loss, assi, iter, n_swaps)
}


/// Perform the initial assignment to medoids
#[inline]
pub(crate) fn initial_assignment<M, N, L, O>(mat: &M, labels: &O, cluster_records: &mut CluRec<usize>) -> (L, Vec<Rec<N>>)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Zero + PartialOrd + Clone + From<N>,
	M: ArrayAdapter<N>,
	O: LabelAdapter<usize>,
{
	let (n, k) = (mat.len(), cluster_records.len());
	assert!(mat.is_square(), "Dissimilarity matrix is not square");
	assert!(n <= u32::MAX as usize, "N is too large");
	assert!(k > 0 && k < u32::MAX as usize, "invalid N");
	assert!(k <= n, "k must be at most N");
	let mut data = vec![Rec::<N>::empty(); mat.len()];

	let loss = data
		.iter_mut()
		.enumerate()
		.map(|(i, cur)| {
			*cur = Rec::new(u32::MAX, N::zero(), u32::MAX, N::zero());
			for (m, &me) in cluster_records.meds.iter().enumerate() {
				if !is_valid_sec_pair(i, labels, m, cluster_records){
					continue;
				}
				let d = mat.get(i, me);
				if i == me || (d < cur.near.d && is_valid_pair(i, labels, m, cluster_records)) {
					cur.seco = cur.near.clone();
					cur.near = DistancePair { i: m as u32, d };
				} else if cur.seco.i == u32::MAX || d < cur.seco.d {
					cur.seco = DistancePair { i: m as u32, d };
				}
				if labels.get(i)!=0{
					cluster_records.label_counts[cur.near.i as usize] += 1;
				}
			}
			L::from(cur.near.d.clone())
		})
		.reduce(L::add)
		.unwrap();
	cluster_records.update_cluster_per_label();
	(loss, data)
}

/// Find the best swap for object j - FastPAM version
#[inline]
pub(crate) fn find_best_swap<M, N, L, O>(
	mat: &M,
	labels: &O,
	removal_loss: &[L],
	data: &[Rec<N>],
	cluster_records: &CluRec<usize>,
	j: usize,
) -> (usize, usize, L)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Signed + Zero + PartialOrd + Clone + From<N> + Copy,
	M: ArrayAdapter<N>,
	O: LabelAdapter<usize>,
{
	let j_label = labels.get(j);
	let mut ploss = removal_loss.to_vec();
	let mut closs = vec![L::zero(); cluster_records.len()];
	let mut acc = vec![L::zero(); cluster_records.no_labels()];
	for (o, reco) in data.iter().enumerate() {
        let o_label = labels.get(o);

		// Skip object and medoid are not label compatible
		if !(j_label == o_label || j_label == 0 || o_label == 0){
			continue;
		}
		let sec_valid = reco.seco.i != u32::MAX;
		let doj = mat.get(o, j);
		// there is an alternative medoid for o
		if sec_valid {
			if doj < reco.near.d {
				if o_label != 0 {
					closs[reco.near.i as usize] += L::from(reco.near.d.clone()) - L::from(reco.seco.d.clone());
				} else {
					ploss[reco.near.i as usize] += L::from(reco.near.d.clone()) - L::from(reco.seco.d.clone());
				}
				acc[o_label] += L::from(doj) - L::from(reco.near.d.clone());
			} else if doj < reco.seco.d {
				// object with label
				if o_label != 0 {
					closs[reco.near.i as usize] += L::from(doj) - L::from(reco.seco.d.clone());
				} else {
					ploss[reco.near.i as usize] += L::from(doj) - L::from(reco.seco.d.clone());
				}
			} 
		// no alternative medoid for o
		} else {
			let benefit = L::from(doj.clone()) - L::from(reco.near.d.clone());

			if benefit < L::zero() {
				ploss[reco.near.i as usize] += benefit;
				acc[o_label] += benefit;
			} else {
				ploss[reco.near.i as usize] += L::from(doj.clone());
			}
		}
	}
	return find_color_min(j, j_label, cluster_records, &mut ploss, &closs, &acc);
}

/// Update the loss when removing each medoid
pub(crate) fn update_removal_loss<N, L, M,  O>(data: &mut [Rec<N>], loss: &mut [L], mat: &M, labels: &O, cluster_records: &mut CluRec<usize>)
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Signed + Clone + Zero + From<N>,
	O: LabelAdapter<usize>,
	M: ArrayAdapter<N>,
{
	loss.fill(L::zero()); // stable since 1.50
	for (i, rec) in data.iter_mut().enumerate() {
		if rec.seco.i == u32::MAX {
			if can_uncolor(cluster_records, labels.get(i)){
				rec.seco = update_second_nearest(mat, cluster_records, labels, rec.near.i as usize, u32::MAX as usize, i, N::zero());
			}
		} else if !is_valid_sec_pair(i, labels, rec.seco.i as usize, cluster_records){
			rec.seco = update_second_nearest(mat, cluster_records, labels, rec.near.i as usize, u32::MAX as usize, i, N::zero());
		}
		loss[rec.near.i as usize] += L::from(rec.seco.d.clone()) - L::from(rec.near.d.clone());
		// as N might be unsigned
	}
}

/// Update the second nearest medoid information
/// Called after each swap.
#[inline]
pub(crate) fn update_second_nearest<M, N, O>(
	mat: &M,
	cluster_records: &CluRec<usize>,
	labels: &O,
	n: usize,
	b: usize,
	o: usize,
	doj: N,
) -> DistancePair<N>
where
	N: Zero + PartialOrd + Clone,
	M: ArrayAdapter<N>,
	O: LabelAdapter<usize>,
{
	// alternatively make sure that doj is 0 when b invalid?
	let mut s = DistancePair::new(b as u32,if b == u32::MAX as usize {
			 N::zero()} 
		else {
			doj.clone()
		});
	for (i, &mi) in cluster_records.meds.iter().enumerate() {
		if i == n || i == b || !is_valid_sec_pair(o, labels, i, cluster_records){
			continue;
		}
		let d = mat.get(o, mi);
		if d < s.d {
			s = DistancePair::new(i as u32, d);
		}
	}
	s
}

/// Perform a single swap
#[inline]
pub(crate) fn do_swap<M, N, L, O>(
	mat: &M,
	labels: &O,
	data: &mut [Rec<N>], 
	cluster_records: &mut CluRec<usize>,
	j: usize,
	b: usize, 
	l: usize,
) -> L
where
	N: Zero + PartialOrd + Clone,
	L: AddAssign + Signed + Zero + PartialOrd + Clone + From<N>,
	M: ArrayAdapter<N>,
	O: LabelAdapter<usize>,
{
	let n = mat.len();
	assert!(b < cluster_records.meds.len(), "invalid medoid number");
	assert!(j < n, "invalid object number");
	cluster_records.meds[b] = j;
	cluster_records.clus_labels[b] = l;
	data.iter_mut()
		.enumerate()
		.map(|(o, reco)| {
			// new medoid
			if o == j {
				if reco.near.i != b as u32 {
					if labels.get(j) != 0 {
						cluster_records.label_counts[reco.near.i as usize] -= 1;
						cluster_records.label_counts[b] += 1;
					}
					reco.seco = reco.near.clone();
				}
				reco.near = DistancePair::new(b as u32, N::zero());
				return L::zero();
			}
			let doj = mat.get(o, j);
			let obj_color = labels.get(o);
			let valid_pair = is_valid_pair(o, labels, b, cluster_records);
			// Nearest medoid is gone:
			if reco.near.i == b as u32 {
				if doj < reco.seco.d || reco.seco.i == u32::MAX{
					reco.near = DistancePair::new(b as u32, doj);
				} else {
					if obj_color != 0 {
						cluster_records.label_counts[reco.near.i as usize] -= 1;
						cluster_records.label_counts[b] += 1;
					}
					reco.near = reco.seco.clone();
					reco.seco = update_second_nearest(mat, &cluster_records, labels, reco.near.i as usize, if valid_pair{b} else {u32::MAX as usize}, o, if valid_pair {doj} else {N::zero()});
				}
			} else {
				// nearest not removed
				if doj < reco.near.d && valid_pair{
					if obj_color != 0 {
						cluster_records.label_counts[reco.near.i as usize] -= 1;
						cluster_records.label_counts[b] += 1;
					}
					if reco.near.d < reco.seco.d || reco.seco.i == u32::MAX{
						reco.seco = reco.near.clone();
					}
					reco.near = DistancePair::new(b as u32, doj);
				} else if doj < reco.seco.d && valid_pair{
					reco.seco = DistancePair::new(b as u32, doj);
				} else if reco.seco.i == b as u32 {
					// second nearest was replaced
					reco.seco = update_second_nearest(mat, cluster_records, labels,  reco.near.i as usize, if valid_pair{b} else {u32::MAX as usize}, o, if valid_pair {doj} else {N::zero()});
				}
			}
			L::from(reco.near.d.clone())
		})
		.reduce(L::add)
		.unwrap()
}

#[inline]
pub(crate)  fn is_valid_pair<O>(obj:usize, labels:&O, med:usize, cluster_records: &CluRec<usize>) -> bool
where 
	O: LabelAdapter<usize>
{
	if labels.get(obj) == 0 || labels.get(obj) == cluster_records.clus_labels[med]{
		return true
	}
	return false
}

#[inline]
pub(crate) fn is_valid_sec_pair<O>(obj:usize, labels:&O, med:usize, cluster_records: &CluRec<usize>) -> bool
where 
	O: LabelAdapter<usize>
{
	if cluster_records.label_counts[med] == 0{
		return true
	} 
	if labels.get(obj) == 0 || labels.get(obj) == cluster_records.clus_labels[med]{
		return true
	}
	return false
}

#[inline]
pub(crate) fn can_uncolor<>(cluster_records: &CluRec<usize>, label:usize) -> bool
{
	if cluster_records.len() == cluster_records.no_labels(){
		return false;
	}
	if cluster_records.clusters_per_label[label] > 1{
		return true;
	}
	return cluster_records.clusters_per_label[0] > 0;
}

#[cfg(test)]
mod tests {
	// TODO: use a larger, much more interesting example.
	use crate::{arrayadapter::LowerTriangle, labeledpam, silhouette, util::assert_array, arrayadapter::LabelList};

	#[test]
	fn testlabeledpam_simple() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let labels = LabelList{
			data: vec![0, 0, 0, 1, 1],
		};
		let mut meds = vec![0, 1];
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = labeledpam(&data, &labels, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 4, "loss not as expected");
		assert_eq!(n_swap, 2, "swaps not as expected");
		assert_eq!(n_iter, 2, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 1, 1], "assignment not as expected");
		assert_array(meds, vec![0, 3], "medoids not as expected");
		assert_eq!(sil, 0.7522494172494172, "Silhouette not as expected");
	}

	#[test]
	fn testlabeledpam_single_cluster() {
		let data = LowerTriangle {
			n: 5,
			data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1],
		};
		let labels = LabelList{
			data: vec![0, 0, 0, 1, 1],
		};
		let mut meds = vec![1]; // So we need one swap
		let (loss, assi, n_iter, n_swap): (i64, _, _, _) = labeledpam(&data, &labels, &mut meds, 10);
		let (sil, _): (f64, _) = silhouette(&data, &assi, false);
		assert_eq!(loss, 14, "loss not as expected");
		assert_eq!(n_swap, 1, "swaps not as expected");
		assert_eq!(n_iter, 1, "iterations not as expected");
		assert_array(assi, vec![0, 0, 0, 0, 0], "assignment not as expected");
		assert_array(meds, vec![0], "medoids not as expected");
		assert_eq!(sil, 0., "Silhouette not as expected");
	}
}
