use std::{fmt::{Debug, Display}, ops::AddAssign, usize};

use num_traits::{AsPrimitive, One, Signed, Zero};

macro_rules! trait_combiner {
    ($combination_name: ident $([$($g: tt: $gc1: tt $(+ $gcn: tt)*),+])? $(: $t: tt $(+ $ts: tt)*)?) => {
        pub trait $combination_name$(<$($g: $gc1 $(+ $gcn)*,)+>)? $(: $t $(+ $ts)*)? {}
        impl<$($($g: $gc1 $(+ $gcn)*,)+)?T $(: $t $(+ $ts)*)?> $combination_name$(<$($g,)+>)? for T {}
    };
}
// Trait for labels 
trait_combiner!(Label: Zero + One + Signed + Max + PartialOrd + Clone + Copy + IntoIndex + FromIndex + Debug + Display);

pub trait Max {
    const MAX: Self;
}
macro_rules! max {
    ($($type:ident),*) => {
        $(impl Max for $type {
            const MAX: Self = <$type>::MAX;
        })*
    };
}

max!(u16, u32, u64, i16, i32, i64);

pub trait IntoIndex:Zero + PartialOrd + Clone + Copy + One{
    /// # Panics
    /// This will panic if the resulting value won't fit into a `usize`, typically because it
    /// contains a fraction or is outside of the range supported by `usize`.
    
    fn into_index(self) -> usize;

    const MAX: Self;
}

macro_rules! into_index {
    ($($type:ident),*) => {
        $(impl IntoIndex for $type {
            fn into_index(self) -> usize {
                self as usize
            }
            const MAX: Self = <$type>::MAX;
        })*
    };
}

into_index!(u16, u32, u64, i16, i32, i64);

pub trait FromIndex {
    fn from_index(index: usize) -> Self;
}

impl<T: Copy+'static> FromIndex for T where usize: AsPrimitive<T>{
    fn from_index(index: usize) -> T {
        index.as_()
    }
}
//Trait  for distances
trait_combiner!(Distance: Zero + PartialOrd + Clone + Debug);

// Trait for loss
trait_combiner!(Loss[N:Distance]: AddAssign + Signed + Zero + PartialOrd + Clone + Display + FiniteAccuracy + (From<N>));

pub trait FiniteAccuracy {
	fn eps() -> Self;
	fn slightly_smaller(&self) -> Self;
	fn slightly_larger(&self) -> Self;
}
macro_rules! int_acc {
    ($($type:ident),*) => {
        $(impl FiniteAccuracy for $type {
            fn eps() -> Self { Self::zero() }
			fn slightly_smaller(&self) -> Self {*self}
			fn slightly_larger(&self) -> Self {*self}
        })*
    };
}
int_acc!(u16, u32, u64, i16, i32, i64);
impl FiniteAccuracy for f32 {
	fn eps() -> Self { 1e-6f32 }
	fn slightly_smaller(&self) -> Self { self / (Self::one() + Self::eps()) }
	fn slightly_larger(&self) -> Self { self * (Self::one() + Self::eps()) }
}
impl FiniteAccuracy for f64 {
	fn eps() -> Self { 1e-12f64 }
	fn slightly_smaller(&self) -> Self { self / (Self::one() + Self::eps()) }
	fn slightly_larger(&self) -> Self { self * (Self::one() + Self::eps()) }
}

