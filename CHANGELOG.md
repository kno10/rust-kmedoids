# Changelog

## kmedoids 0.1.4 (2021-01-31)

- remove forgotten println in pam reference method

## kmedoids 0.1.3 (2021-01-31)

- improve ndarray adapter to accept views
- relax version dependencies slightly

## kmedoids 0.1.2 (2020-12-24)

- reordered the returned values by importance to end users
- add FastPAM1 (slower, but exact same result as PAM)
- add original PAM (much slower, for reference)
- PAM BUILD and SWAP independently

## kmedoids 0.1.1 (2020-12-23)

- allow use with different array types (ndarray, lower triangular in a vec)

## kmedoids 0.1.0 (2020-12-22)

- initial port to Rust of the core FasterPAM functionality
- first Rust program ever, hence there likely is room for improvement
