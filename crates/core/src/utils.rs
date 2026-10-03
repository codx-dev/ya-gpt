#[inline(always)]
pub fn sqrtf(a: f32) -> f32 {
    #[cfg(not(feature = "std"))]
    return libm::sqrtf(a);

    #[cfg(feature = "std")]
    return a.sqrt();
}

#[inline(always)]
pub fn expf(a: f32) -> f32 {
    #[cfg(not(feature = "std"))]
    return libm::expf(a);

    #[cfg(feature = "std")]
    return a.exp();
}

#[inline(always)]
pub fn logf(a: f32) -> f32 {
    #[cfg(not(feature = "std"))]
    return libm::logf(a);

    #[cfg(feature = "std")]
    return a.ln();
}

#[inline(always)]
pub fn powi(a: f32, i: i32) -> f32 {
    #[cfg(not(feature = "std"))]
    return libm::powf(a, i as f32);

    #[cfg(feature = "std")]
    return a.powi(i);
}
