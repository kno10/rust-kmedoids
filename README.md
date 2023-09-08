# k-Medoids Clustering in Rust with FasterPAM

This Rust crate implements k-medoids clustering with PAM and variants of clustering by direct optimization of the (Medoid) Silhouette.
It can be used with arbitrary dissimilarites, as it requires a dissimilarity matrix as input.

This software package has been introduced in JOSS:

> Erich Schubert and Lars Lenssen  
> **Fast k-medoids Clustering in Rust and Python**  
> Journal of Open Source Software 7(75), 4183  
> <https://doi.org/10.21105/joss.04183> (open access)

For further details on the implemented algorithm FasterPAM, see:

> Erich Schubert, Peter J. Rousseeuw  
> **Fast and Eager k-Medoids Clustering:**  
> **O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms**  
> Information Systems (101), 2021, 101804  
> <https://doi.org/10.1016/j.is.2021.101804> (open access)

an earlier (slower, and now obsolete) version was published as:

> Erich Schubert, Peter J. Rousseeuw:  
> **Faster k-Medoids Clustering: Improving the PAM, CLARA, and CLARANS Algorithms**  
> In: 12th International Conference on Similarity Search and Applications (SISAP 2019), 171-187.  
> <https://doi.org/10.1007/978-3-030-32047-8_16>  
> Preprint: <https://arxiv.org/abs/1810.05691>

This is a port of the original Java code from [ELKI](https://elki-project.github.io/) to Rust.

For further details on medoid Silhouette clustering with automatic cluster number selection (FasterMSC, DynMSC), see:

> Lars Lenssen, Erich Schubert:  
> **Medoid silhouette clustering with automatic cluster number selection** 
> Preprint: <https://arxiv.org/abs/2309.03751>

an earlier version was published as:

> Lars Lenssen, Erich Schubert:  
> **Clustering by Direct Optimization of the Medoid Silhouette**  
> In: 15th International Conference on Similarity Search and Applications (SISAP 2022)  
> <https://doi.org/10.1007/978-3-031-17849-8_15>

If you use this code in scientific work, please cite above papers. Thank you.


## Example

```
let dissim = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
let (loss, assingment, n_iter, n_swap): (f64, _, _, _) = kmedoids::fasterpam(&dissim, &mut meds, 100);
println!("Loss is: {}", loss);
```

Note that:

* you need to specify the "output" data type of `loss` -- chose a signed type with sufficient precision.
For example for unsigned distances using `u32`, it may be better to use `i64` to compute the loss.
* the input distance type needs to be convertible into the output data type via `Into`


## Implemented Algorithms

* **FasterPAM** (Schubert and Rousseeuw, 2020, 2021)
* FasterPAM with an integrated additional shuffling step
* Parallelized FasterPAM with an integrated additional shuffling step
* FastPAM1 (Schubert and Rousseeuw, 2019, 2021)
* PAM (Kaufman and Rousseeuw, 1987) with BUILD and SWAP
* Alternating optimization (k-means-style algorithm)
* Silhouette index for evaluation (Rousseeuw, 1987)
* **FasterMSC** (Lenssen and Schubert, 2022)
* FastMSC (Lenssen and Schubert, 2022)
* DynMSC (Lenssen and Schubert, 2023)
* PAMSIL (Van der Laan and Pollard, 2003)
* PAMMEDSIL (Van der Laan and Pollard, 2003)

Note that the k-means-like algorithm for k-medoids tends to find much worse solutions.

The additional shuffling step for FasterPAM is beneficial if you intend to restart
k-medoids multiple times on the same data (to find better solutions).
The parallel implementation is typically faster when you have more than 5000 instances.

## Rust Dependencies

* [num-traits](https://docs.rs/num-traits/) for supporting different numeric types
* [ndarray](https://docs.rs/ndarray/) for arrays (optional)
* [rand](https://docs.rs/rand/) for random initialization (optional)
* [rayon](https://docs.rs/rayon/) for parallelization (optional)

## Contributing to `rust-kmedoids`

Third-party contributions are welcome. Please use [pull requests](https://github.com/kno10/rust-kmedoids/pulls) to submit patches.

## Reporting issues

Please report errors as an [issue](https://github.com/kno10/rust-kmedoids/issues) within the repository's issue tracker.

## Support requests

If you need help, please submit an [issue](https://github.com/kno10/rust-kmedoids/issues) within the repository's issue tracker.

## License: GPL-3 or later

> This program is free software: you can redistribute it and/or modify
> it under the terms of the GNU General Public License as published by
> the Free Software Foundation, either version 3 of the License, or
> (at your option) any later version.
> 
> This program is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
> GNU General Public License for more details.
> 
> You should have received a copy of the GNU General Public License
> along with this program.  If not, see <https://www.gnu.org/licenses/>.
