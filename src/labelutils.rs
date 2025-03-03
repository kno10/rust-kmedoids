use std::{fmt::{Debug, Display}, usize};

use crate::types::{Distance, Label, Loss};




#[allow(clippy::len_without_is_empty)]
pub trait LabelAdapter<N>:{
	/// Get the length of an array structure
	fn len(&self) -> usize;
	/// Get the contents at cell x,y
	fn get(&self, x: usize) -> &N;
	fn iter<'a,'b>(&'b self) -> impl Iterator<Item = &'a N> where 'b: 'a, N: 'a;
}

/// Adapter trait for using `ndarray::Array2` and similar
#[cfg(feature = "ndarray")]
impl<A, N> LabelAdapter<N> for ndarray::ArrayBase<A, ndarray::Ix1>
where
	A: ndarray::Data<Elem = N>,
	N: Clone,
{
	#[inline]
	fn len(&self) -> usize {
		self.shape()[0]
	}
	#[inline]
	fn get(&self, x: usize) -> &N {
		self.get(x).unwrap()
	}
	#[inline]
	fn iter<'a,'b>(&'b self) -> impl Iterator<Item = &'a N> where 'b: 'a, N: 'a {
		self.iter()
	}
}

pub struct LabelList<C> {
	pub data: Vec<C>,
}

impl <C: Label> LabelAdapter<C> for LabelList<C> {
	#[inline]
	fn len(&self) -> usize {
		self.data.len()
	}
	#[inline]
	fn get(&self, x: usize) -> &C {
		self.data.get(x).unwrap()
	}
	#[inline]
	fn iter<'a,'b>(&'b self) -> impl Iterator<Item = &'a C> where 'b: 'a, C: 'a {
		self.data.iter()
	}
}

/// Information kept about the clustering
#[derive(Debug)]
pub(crate) struct ClusterRecords<'a, C> 
where
	C: Label,
{
	pub(crate) medoid_index_map: & 'a mut [usize],
	pub(crate) cluster_labels: Vec<C>,
	pub(crate) label_counts: Vec<usize>,
	pub(crate) clusters_per_label: Vec<usize>,
	pub(crate) unlabeled_clusters: usize,
}

impl<'a, C:Label> ClusterRecords<'a, C> 
{
	pub(crate) fn new(meds: &'a mut [usize], labels: &impl LabelAdapter<C>, label_count:usize) -> ClusterRecords<'a, C>
	{
		let meds_len = meds.len();
		let clus_labels:Vec<C> = meds.iter().map(|m|{
			*labels.get(*m)
		}).collect();
		let mut clu_rec = ClusterRecords {
			medoid_index_map: meds,
			cluster_labels: clus_labels,
			label_counts: vec![0; meds_len],
			clusters_per_label: vec![0; label_count],
			unlabeled_clusters: 0,
		};
		clu_rec.update_cluster_per_label();
		return clu_rec;
	}

	pub(crate) fn len(&self) -> usize {
		self.medoid_index_map.len()
	}

	pub(crate) fn no_labels(&self) -> usize {
		self.clusters_per_label.len()
	}

	// update all the label information for the clusters
	pub(crate) fn update_labels(&mut self){
		// reset labels if no labeled point is present
		for (i, cluster_label) in self.cluster_labels.iter_mut().enumerate(){
			if self.label_counts[i] == 0{
				*cluster_label = -C::one();
			}
		}
		self.update_cluster_per_label()
	}

	// update the number of clusters per label
	#[inline(always)]
	pub(crate) fn update_cluster_per_label(&mut self){
		self.clusters_per_label.fill(0);
		self.unlabeled_clusters = 0;
		for label in self.cluster_labels.iter(){
			if *label >= C::zero(){
				self.clusters_per_label[label.into_index()] += 1;
			} else {
				self.unlabeled_clusters += 1;
			}
		}
	}
}

impl <'a, C> Display for ClusterRecords<'a, C>
where
	C: Label,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "CluRec:\n medoids: {:?},\n clu_lab: {:?},\n lab_cnt: {:?},\n cl_p_la: {:?},\n unl_clu: {}", self.medoid_index_map, self.cluster_labels, self.label_counts, self.clusters_per_label, self.unlabeled_clusters)
	}
}

#[inline]
pub(crate) fn can_uncolor<C: Label>(cluster_records: &ClusterRecords<C>, label:C) -> bool
{
	cluster_records.len() > cluster_records.no_labels() && (cluster_records.unlabeled_clusters > 0 || (label >= C::zero() && cluster_records.clusters_per_label[label.into_index()] > 1))
}

// find the labeled min (index, value and label)
#[inline]
pub(crate) fn find_label_min<'a, N, L, C>(med_label:C, cluster_records:&ClusterRecords<C>, losses:&mut Vec<L>, colored_losses:&Vec<L>, acc:L, colored_acc:&Vec<L>) -> (usize, C, L)
	where
		N: Distance,
		L: Loss<N>,
		C: Label,
{
	let mut base_label = med_label;
	let mut alt_label = -C::one();
	let mut min = L::zero();
	let mut min2 = L::zero();
	let mut best_index = usize::MAX;
	let mut best_label:C = -C::one();
	let mut best_loss = L::eps();
	// if the candidate is not labeled, find the best two colors
	if base_label == -C::one() {
		for (i, a) in colored_acc.iter().enumerate() {
			if *a < min {
				min2 = min;
				alt_label = base_label;
				min = a.clone();
				base_label = C::from_index(i);
			} else if *a < min2 {
				min2 = a.clone();
				alt_label = C::from_index(i);
			}
		}
	}
	for (i_med, med_loss) in losses.iter().enumerate() {
		let cluster_label = cluster_records.cluster_labels[i_med];
		if med_label >= C::zero() {
			debug_assert!(med_label == base_label, "Incompatible labels");
			if med_label != cluster_label && !can_uncolor(cluster_records, cluster_label){
				// labels incompatible
				continue;
			} else if base_label == cluster_label {
				debug_assert!(med_label == cluster_label, "Incompatible labels");
				let temp_loss = med_loss.clone() + colored_losses[i_med].clone() + acc.clone() + colored_acc[cluster_label.into_index()].clone();
				if temp_loss < best_loss {
					best_loss = temp_loss;
					best_index = i_med;
					best_label = base_label;
				}
			} else {
				debug_assert!(med_label == base_label, "Incompatible labels");
				let temp_loss = med_loss.clone() + acc.clone() + colored_acc[med_label.into_index()].clone();
				if temp_loss < best_loss {
					best_loss = temp_loss;
					best_index = i_med;
					best_label = base_label;
				}
			}
		} else if can_uncolor(cluster_records, cluster_label) {
			let mut local_best = base_label;
			if local_best != cluster_label && colored_losses[i_med] < L::zero(){
				local_best = if min < colored_acc[cluster_label.into_index()].clone() + colored_losses[i_med].clone() {local_best} else {cluster_label};
			}else if local_best == cluster_label && colored_losses[i_med] > L::zero(){
				local_best = if colored_acc[local_best.into_index()].clone() + colored_losses[i_med].clone() < min2 {local_best} else {alt_label};
			}
			let mut temp_loss = med_loss.clone() + acc.clone();
			if local_best == cluster_label {
				temp_loss += colored_losses[i_med].clone();
			}
			// add the benefit of the best label
			if local_best >= C::zero() {
				temp_loss += colored_acc[local_best.into_index()].clone();
			}
			if temp_loss < best_loss {
				best_loss = temp_loss;
				best_index = i_med;
				best_label = local_best;
			}
		} else {
			// cluster can not be relabeled
			// cluster can not be unlabeled
			debug_assert!(cluster_label >= C::zero(), "Cluster does not have a valid label but also does not have an alternative cluster");
			let temp_loss = med_loss.clone() + colored_losses[i_med].clone() + acc.clone() + colored_acc[cluster_label.into_index()].clone();
			if temp_loss < best_loss {
				best_loss = temp_loss;
				best_index = i_med;
				best_label = cluster_label;
			}
		}
	}
	debug_assert!(best_label == cluster_records.cluster_labels[best_index] || can_uncolor(cluster_records, cluster_records.cluster_labels[best_index]));
	return (best_index, best_label, best_loss);
}