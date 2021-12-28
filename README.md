# k-Medoids Clustering in Rust with FasterPAM

This Rust crate implements k-medoids clustering with PAM.
It can be used with arbitrary dissimilarites, as it requires a dissimilarity matrix as input.

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

If you use this code in scientific work, please cite above papers. Thank you.


## Example

```
let dissim = ndarray::arr2(&[[0,1,2,3],[1,0,4,5],[2,4,0,6],[3,5,6,0]]);
let mut meds = kmedoids::random_initialization(4, 2, &mut rand::thread_rng());
let (loss, assingment, n_iter, n_swap): (f64, _, _, _) = kmedoids::fasterpam(&dissim, &mut meds, 100).unwrap();
println!("Loss is: {}", loss);
```

Note that:

* you need to specify the "output" data type of `loss` -- chose a signed type with sufficient precision.
For example for unsigned distances using `u32`, it may be better to use `i64` to compute the loss.
* the function can return an error if a conversion of the input data to the output is not possible (e.g., when converting a very large integer into a low precision integer loss).

## Implemented Algorithms

* **FasterPAM** (Schubert and Rousseeuw, 2020, 2021)
* FastPAM1 (Schubert and Rousseeuw, 2019, 2021)
* PAM (Kaufman and Rousseeuw, 1987) with BUILD and SWAP
* Alternating optimization (k-means-style algorithm)

Note that the k-means-like algorithm tends to find much worse solutions.

## Rust Dependencies

* [num-traits](https://docs.rs/num-traits/) for supporting different numeric types
* [ndarray](https://docs.rs/ndarray/) for arrays (optional)
* [rand](https://docs.rs/rand/) for random initialization (optional)

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

## FAQ: Why GPL and not Apache/MIT/BSD?

Because copyleft software like Linux is what built the open-source community.

Tit for tat: you get to use my code, I get to use your code.
