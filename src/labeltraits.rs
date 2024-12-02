use num_traits::{AsPrimitive, One, Zero};


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

// macro_rules! from_index {
//     ($($type:ident),*) => {
//         $(impl FromIndex for $type {
//             fn from_index(index: usize) -> $type {
//                 index as $type
//             }
//         })*
//     };
// }

// from_index!(u16, u32, u64, i16, i32, i64);