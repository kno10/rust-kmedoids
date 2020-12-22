/// Add without overflow, for both integer and float types.
///
/// This is pre-generated for standard numerical types such as i64 and f64.
///
/// You only need to touch this if you want to extend this to additional numerical types.
pub trait SafeAdd {
	/// The `return a + b` addition operation
	fn safe_add(self, o: Self) -> Self;
	/// The `a += b` increment operation
	fn safe_inc(&mut self, o: Self);
}

macro_rules! safeadd_integer_impl {
	($trait_name:ident for $($t:ty)*) => {$(
		impl $trait_name for $t {
			#[inline]
			fn safe_add(self, o: Self) -> Self {
				return self.saturating_add(o);
			}
			#[inline]
			fn safe_inc(&mut self, o: Self) {
				*self = self.saturating_add(o);
			}
		}
	)*}
}
macro_rules! safeadd_float_impl {
	($trait_name:ident for $($t:ty)*) => {$(
		impl $trait_name for $t {
			#[inline]
			fn safe_add(self, o: Self) -> Self {
				return self + o;
			}

			#[inline]
			fn safe_inc(&mut self, o: Self) {
				*self += o;
			}
		}
	)*}
}
safeadd_integer_impl!(SafeAdd for isize usize i8 u8 i16 u16 i32 u32 i64 u64);
#[cfg(has_i128)]
safeadd_integer_impl!(SafeAdd for i128 u128);
safeadd_float_impl!(SafeAdd for f32 f64);
