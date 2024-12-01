use crate::{arrayadapter::ArrayAdapter, can_uncolor};
use core::ops::AddAssign;
use num_traits::{Signed, ToPrimitive, Zero};

/// Object id and distance pair
#[derive(Debug, Copy, Clone)]
pub(crate) struct DistancePair<N> {
	pub(crate) i: u32,
	pub(crate) d: N,
}
impl<N> DistancePair<N> {
	pub(crate) fn new(i: u32, d: N) -> Self {
		DistancePair { i, d }
	}
}
impl<N: Zero> DistancePair<N> {
	pub(crate) fn empty() -> Self {
		DistancePair {
			i: u32::MAX,
			d: N::zero(),
		}
	}
}

/// Information kept for each point: two such pairs
#[derive(Debug, Copy, Clone)]
pub(crate) struct Rec<N> {
	pub(crate) near: DistancePair<N>,
	pub(crate) seco: DistancePair<N>,
}
impl<N> Rec<N> {
	pub(crate) fn new(i1: u32, d1: N, i2: u32, d2: N) -> Rec<N> {
		Rec {
			near: DistancePair { i: i1, d: d1 },
			seco: DistancePair { i: i2, d: d2 },
		}
	}
}
impl<N: Zero> Rec<N> {
	pub(crate) fn empty() -> Self {
		Rec {
			near: DistancePair::empty(),
			seco: DistancePair::empty(),
		}
	}
}

/// Information kept about the clustering
#[derive(Debug)]
pub(crate) struct CluRec<'a, N> 
where
	N: Zero + PartialOrd + Clone,
{
	pub(crate) meds: & 'a mut [usize],
	pub(crate) clus_labels: Vec<N>,
	pub(crate) label_counts: Vec<usize>,
	pub(crate) clusters_per_label: Vec<usize>,
	//pub(crate) unlabeled_clusters: usize,
}

impl<'a, N:Zero + PartialOrd + Clone + ToPrimitive> CluRec<'a, N> 
{
	pub(crate) fn new(meds: &mut [usize], label_count:usize) -> CluRec<N>
	{
		let meds_len = meds.len();
		CluRec {
			meds,
			clus_labels: vec![N::zero(); meds_len],
			label_counts: vec![usize::zero(); meds_len],
			clusters_per_label: vec![usize::zero(); label_count + 1],
			//unlabeled_clusters: 0,
		}
	}

	pub(crate) fn len(&self) -> usize {
		self.meds.len()
	}

	pub(crate) fn no_labels(&self) -> usize {
		self.clusters_per_label.len()
	}

	pub(crate) fn update_cluster_per_label(&mut self){
		self.clusters_per_label.fill(usize::zero());
		for label in self.clus_labels.iter(){
			self.clusters_per_label[label.to_usize().unwrap()] += 1;
		}
	}
}


/// Information kept for each point: three such pairs
#[derive(Debug, Copy, Clone)]
pub(crate) struct Reco<N> {
	pub(crate) near: DistancePair<N>,
	pub(crate) seco: DistancePair<N>,
	pub(crate) third: DistancePair<N>,
}
impl<N> Reco<N> {
	pub(crate) fn new(i1: u32, d1: N, i2: u32, d2: N, i3: u32, d3: N) -> Reco<N> {
		Reco {
			near: DistancePair { i: i1, d: d1 },
			seco: DistancePair { i: i2, d: d2 },
			third: DistancePair { i: i3, d: d3 },
		}
	}
}
impl<N: Zero> Reco<N> {
	pub(crate) fn empty() -> Self {
		Reco {
			near: DistancePair::empty(),
			seco: DistancePair::empty(),
			third: DistancePair::empty(),
		}
	}
}

/// Find the minimum (index and value)
#[inline]
pub(crate) fn find_min<'a, L, I>(a: &mut I) -> (usize, L)
	where
		L: PartialOrd + Clone + Zero + 'a,
		I: std::iter::Iterator<Item = &'a L> + 'a,
{
	let mut a = a.enumerate();
	let mut best: (usize, L) = (0, a.next().unwrap().1.clone());
	for (ik, iv) in a {
		if *iv < best.1 {
			best = (ik, iv.clone());
		}
	}
	best
}

// find the labeled min (index, value and label)
#[inline]
pub(crate) fn find_color_min<'a, L>(j:usize, j_label:usize, cluster_records:&CluRec<usize>, ploss:&mut Vec<L>, closs:&Vec<L>, acc:&Vec<L>) -> (usize, usize, L)
	where
		L: PartialOrd + Clone + Zero + 'a + AddAssign + Copy,
{
	let mut best_label = j_label;
	let mut second_label = usize::MAX;
	let mut min = L::zero();
	let mut min2 = L::zero();
	let mut best = 0;
	let mut best_color = 0;
	let mut loss = L::zero();
	// if the candidate is not labeled, find the best two colors
	if best_label == 0 {
		for (i, a) in acc.iter().enumerate().skip(1) {
			if *a < min {
				min2 = min;
				second_label = best_label;
				min = *a;
				best_label = i;
			} else if *a < min2 {
				min2 = *a;
				second_label = i;
			}
		}
	}
	for (i, i_loss) in ploss.iter().enumerate() {
		let cluster_label = cluster_records.clus_labels[j];
		if j_label != 0 {
			if j_label != cluster_label && !can_uncolor(cluster_records, cluster_label){
				// labels incompatible
				continue;
			} else if best_color == cluster_label {
				let temp_loss = *i_loss + closs[i] + acc[0] + acc[cluster_label];
				if temp_loss < loss {
					loss = temp_loss;
					best = i;
					best_color = best_color;
				}
			} else {
				let temp_loss = *i_loss + acc [0] + acc[cluster_label];
				if temp_loss < loss {
					loss = temp_loss;
					best = i;
					best_color = best_color;
				}
			}
		} else if can_uncolor(cluster_records, cluster_label) {
			let mut local_best = best_label;
			if closs[i] < L::zero(){
				local_best = if min < acc[cluster_label] + closs[i] {local_best} else {cluster_label};
			}else if local_best == cluster_label && closs[i] > L::zero(){
				local_best = if acc[local_best] + closs[i] < min2 {local_best} else {second_label};
			}
			let mut temp_loss = *i_loss + acc[0];
			if local_best == cluster_label {
				temp_loss += closs[i];
			}
			// add the benefit of the best label
			if local_best > 0 {
				temp_loss += acc[local_best];
			}
			if temp_loss < loss {
				loss = temp_loss;
				best = i;
				best_color = local_best;
			}
		} else {
			// cluster can not be relabeled
			// cluster can not be unlabeled
			let temp_loss = *i_loss + closs[i] + acc[0] + acc[cluster_label];
			if temp_loss < loss {
				loss = temp_loss;
				best = i;
				best_color = cluster_label;
			}
		}
	}
	return (best, best_color, loss);
}

/// Find the maximum (index and value)
#[inline]
pub(crate) fn find_max<'a, L, I>(a: &mut I) -> (usize, L)
	where
		L: PartialOrd + Clone + Zero + 'a,
		I: std::iter::Iterator<Item = &'a L> + 'a,
{
	let mut a = a.enumerate();
	let mut best: (usize, L) = (0, a.next().unwrap().1.clone());
	for (ik, iv) in a {
		if *iv > best.1 {
			best = (ik, iv.clone());
		}
	}
	best
}

/// Choose the best medoid within a partition
/// Used by ther alternating algorithm, or when a single cluster is requested.
pub(crate) fn choose_medoid_within_partition<M, N, L>(
	mat: &M,
	assi: &[usize],
	med: &mut [usize],
	m: usize,
) -> (bool, L)
	where
		N: PartialOrd + Clone,
		L: AddAssign + Signed + Zero + PartialOrd + Clone + From<N>,
		M: ArrayAdapter<N>,
{
	let first = med[m];
	let mut best = first;
	let mut sumb = L::zero();
	for (i, &a) in assi.iter().enumerate() {
		if first != i && a == m {
			sumb += L::from(mat.get(first, i));
		}
	}
	for (j, &aj) in assi.iter().enumerate() {
		if j != first && aj == m {
			let mut sumj = L::zero();
			for (i, &ai) in assi.iter().enumerate() {
				if i != j && ai == m {
					sumj += L::from(mat.get(j, i));
				}
			}
			if sumj < sumb {
				best = j;
				sumb = sumj;
			}
		}
	}
	med[m] = best;
	(best != first, sumb)
}

/// Debug helper function
pub(crate) fn debug_assert_assignment<M, N>(_mat: &M, _med: &[usize], _data: &[Rec<N>])
	where
		N: PartialOrd + Clone,
		M: ArrayAdapter<N>,
{
	#[cfg(feature = "assertions")]
	for o in 0.._mat.len() {
		debug_assert!(
			_mat.get(o, _med[_data[o].near.i as usize]) == _data[o].near.d,
			"primary assignment inconsistent"
		);
		debug_assert!(
			_mat.get(o, _med[_data[o].seco.i as usize]) == _data[o].seco.d,
			"secondary assignment inconsistent"
		);
		debug_assert!(
			_data[o].near.d <= _data[o].seco.d,
			"nearest is farther than second nearest"
		);
	}
}

/// Debug helper function, for methods with three nearest medoids
pub(crate) fn debug_assert_assignment_th<M, N>(_mat: &M, _med: &[usize], _data: &[Reco<N>])
	where
		N: PartialOrd + Clone,
		M: ArrayAdapter<N>,
{
	#[cfg(feature = "assertions")]
	for o in 0.._mat.len() {
		debug_assert!(
			_mat.get(o, _med[_data[o].near.i as usize]) == _data[o].near.d,
			"primary assignment inconsistent"
		);
		debug_assert!(
			_mat.get(o, _med[_data[o].seco.i as usize]) == _data[o].seco.d,
			"secondary assignment inconsistent"
		);
		debug_assert!(
			_mat.get(o, _med[_data[o].third.i as usize]) == _data[o].third.d,
			"third assignment inconsistent"
		);
		debug_assert!(
			_data[o].near.d <= _data[o].seco.d,
			"nearest is farther than second nearest"
		);
		debug_assert!(
			_data[o].seco.d <= _data[o].third.d,
			"second nearest is farther than third nearest"
		);
	}
}

/// test two arrays for equality, used in tests only
#[cfg(test)]
pub(crate) fn assert_array(result: Vec<usize>, expect: Vec<usize>, msg: &'static str) {
	assert!(
		result.iter().zip(expect.iter()).all(|(a, b)| a == b),
		"{}",
		msg
	);
}
