//! FFT layer, backed by [`rustfft`].
//!
//! The engine only ever needs a 256-point forward transform of a real signal
//! (docs/02-algorithm-spec.md §2), so this module exposes a deliberately small
//! one-sided magnitude API rather than raw complex buffers.
//!
//! # Why a wrapper instead of calling rustfft directly
//!
//! * `rustfft` operates on `Complex<f32>` buffers; the analysis wants a
//!   windowed real frame in and a one-sided magnitude spectrum out. The
//!   adapter keeps that conversion (and the planner reuse) in one place.
//! * The plan is expensive to build and cheap to reuse, so it is created once
//!   per thumbnail and kept alongside a reusable scratch buffer.
//! * It gives the rest of the crate a single seam to swap in a different
//!   backend (e.g. `realfft`) without touching the DSP stages.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// A planned forward FFT for real-valued input of a fixed size.
pub struct RealFft {
    n: usize,
    plan: Arc<dyn Fft<f32>>,
    buffer: Vec<Complex<f32>>,
}

impl RealFft {
    /// Plan a forward transform of size `n`.
    ///
    /// Panics if `n` is not a power of two (the analysis only ever uses 256).
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT size must be a power of two, got {n}");
        let mut planner = FftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(n);
        RealFft {
            n,
            plan,
            buffer: vec![Complex::new(0.0, 0.0); n],
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// One-sided magnitude spectrum of `input` into `out` (`out.len() == n/2 + 1`).
    ///
    /// `out[k] = sqrt(re[k]² + im[k]²)` for `k` in `0..=n/2`.
    ///
    /// The magnitude is computed manually rather than via `Complex::norm()`
    /// because `norm()` uses `hypot`, which is more accurate but slower — and
    /// this is the hot path (one call per 256-sample frame).
    pub fn magnitudes(&mut self, input: &[f32], out: &mut [f32]) {
        let n = self.n;
        debug_assert!(input.len() >= n);
        debug_assert!(out.len() > n / 2);

        for (slot, &x) in self.buffer.iter_mut().zip(input.iter()) {
            *slot = Complex::new(x, 0.0);
        }
        self.plan.process(&mut self.buffer);

        for (k, o) in out.iter_mut().enumerate().take(n / 2 + 1) {
            let c = self.buffer[k];
            *o = (c.re * c.re + c.im * c.im).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dft_mag(x: &[f32]) -> Vec<f32> {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                let mut re = 0f64;
                let mut im = 0f64;
                for (i, &v) in x.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                    re += v as f64 * a.cos();
                    im += v as f64 * a.sin();
                }
                (re * re + im * im).sqrt() as f32
            })
            .collect()
    }

    #[test]
    fn matches_naive_dft() {
        let n = 256;
        let mut fft = RealFft::new(n);
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                (2.0 * std::f32::consts::PI * 5.0 * t).sin()
                    + 0.4 * (2.0 * std::f32::consts::PI * 23.0 * t).cos()
            })
            .collect();
        let mut out = vec![0f32; n / 2 + 1];
        fft.magnitudes(&input, &mut out);
        let want = naive_dft_mag(&input);
        for (k, (a, b)) in out.iter().zip(want.iter()).enumerate() {
            let tol = 1e-3 * (1.0 + b.abs());
            assert!((a - b).abs() < tol, "bin {k}: got {a}, want {b}");
        }
    }

    #[test]
    fn dc_bin_is_the_sum_of_samples() {
        let n = 64;
        let mut fft = RealFft::new(n);
        let input = vec![1.0f32; n];
        let mut out = vec![0f32; n / 2 + 1];
        fft.magnitudes(&input, &mut out);
        assert!((out[0] - n as f32).abs() < 1e-2, "dc = {}", out[0]);
        // A constant signal has no energy in any other bin.
        for (k, v) in out.iter().enumerate().skip(1) {
            assert!(*v < 1e-2, "bin {k} = {v}");
        }
    }

    #[test]
    fn plan_is_reusable_across_frames() {
        let n = 256;
        let mut fft = RealFft::new(n);
        let mut out = vec![0f32; n / 2 + 1];
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let mut first = vec![0f32; n / 2 + 1];
        fft.magnitudes(&a, &mut out);
        first.copy_from_slice(&out);
        // Interleave a different frame, then repeat the first: must be identical.
        fft.magnitudes(&b, &mut out);
        fft.magnitudes(&a, &mut out);
        assert_eq!(first, out);
    }
}
