pub trait IntoIndex {
    /// # Panics
    /// This will panic if the resulting value won't fit into a `usize`, typically because it
    /// contains a fraction or is outside of the range supported by `usize`.
    fn into_index(self) -> usize;
}

macro_rules! into_index {
    ($($type:ident),*) => {
        $(impl IntoIndex for $type {
            fn into_index(self) -> usize {
                self as usize
            }
        })*
    };
}

into_index!(i16, i32, i64);

pub trait FromIndex {
    fn from_index(index: usize) -> Self;
}

macro_rules! from_index {
    ($($type:ident),*) => {
        $(impl FromIndex for $type {
            fn from_index(index: usize) -> $type {
                index as $type
            }
        })*
    };
}

from_index!(i16, i32, i64);