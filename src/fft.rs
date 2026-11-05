//! Bit-exact port of the original's spectral engine FFT.
//!
//! `CPfSpecEngBuilder` (`InitializeFFT` + `RealFFT` + `DoFFTToMagnitude`,
//! re/02 and re/03) is *not* a textbook real FFT: the twiddle table is stored
//! in bit-reversed order, the decimation-in-frequency pass reads one twiddle
//! per 2i-complex block, every butterfly output is halved (the spectrum comes
//! out scaled by 1/128), and the magnitudes are read back through the
//! bit-reversal table. A generic FFT library produces the same spectrum up to
//! that 1/128 factor, but the analyzer's onset ladder thresholds are a
//! discrete ÷1.2 sequence, so even a uniform scale difference can move which
//! rung the crossing lands on and change the detected onset set. This module
//! therefore reproduces the original's arithmetic step for step, in `f32`.
//!
//! The scaling does not affect the rest of the pipeline in any other way:
//! peak-hold, both impact detectors and the chain confidences are linear or
//! normalised, so only the ladder rung is scale-sensitive.

/// The original's spectral-engine transform, built once per analysis.
pub struct SpecEngFft {
    n: usize,
    /// Float-slot index for natural index k (always even).
    bitrev: Vec<usize>,
    /// Twiddle pairs, stored bit-reversed: `(sin(-2πj/N), -cos(2πj/N))` at
    /// float slots `(bitrev[j], bitrev[j]+1)`.
    tw: Vec<f32>,
    /// `f32(cos(2πi/N)·-0.46 + 0.54)`, computed in `f64` like the original.
    window: Vec<f32>,
    /// Scratch: the windowed frame being transformed.
    work: Vec<f32>,
}

impl SpecEngFft {
    /// Build the tables for a forward transform of size `n`.
    ///
    /// Panics if `n` is not a power of two (the analysis only ever uses 256).
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT size must be a power of two, got {n}");
        let half = n / 2;

        // bitrev[k]: each pass shifts the accumulator right and ORs in
        // `half` when the corresponding bit of k is set — the standard
        // bit-reversal of log2(half) bits, doubled.
        let mut bitrev = vec![0usize; half];
        for (k, br) in bitrev.iter_mut().enumerate() {
            let mut v7 = half / 2;
            let mut v6 = 0usize;
            while v7 > 0 {
                let v4 = v6 >> 1;
                let v5 = if k & v7 != 0 { half } else { 0 };
                v6 = v5 + v4;
                v7 >>= 1;
            }
            *br = v6;
        }

        let mut tw = vec![0f32; 2 * half];
        for j in 0..half {
            let ang = -2.0 * std::f64::consts::PI * j as f64 / n as f64;
            tw[bitrev[j]] = ang.sin() as f32;
            tw[bitrev[j] + 1] = -((2.0 * std::f64::consts::PI * j as f64 / n as f64).cos() as f32);
        }

        let window = (0..n)
            .map(|i| {
                let ang = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (ang.cos() * -0.46 + 0.54) as f32
            })
            .collect();

        SpecEngFft {
            n,
            bitrev,
            tw,
            window,
            work: vec![0f32; n],
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `DoFFTToMagnitude`: `n/2` bin magnitudes of one real `n`-sample frame.
    ///
    /// `out[k] = sqrt(re² + im²)` of bin k, with the whole spectrum scaled by
    /// the original's `1/(n/2)` butterfly halving. `out.len() == n/2`.
    pub fn magnitudes(&mut self, input: &[f32], out: &mut [f32]) {
        let n = self.n;
        let half = n / 2;
        debug_assert!(input.len() >= n);
        debug_assert_eq!(out.len(), half);

        let work = &mut self.work;
        for j in 0..n {
            work[j] = input[j] * self.window[j];
        }

        // ── staged DIF pass (RealFFT): stages i = n/4 … 1, blocks of 2i
        //    complex, one twiddle per block read sequentially from tw[0] ──
        let mut i = n >> 2;
        while i > 0 {
            let nblocks = half / (2 * i);
            for b in 0..nblocks {
                let base = b * 2 * i;
                let c = self.tw[2 * b];
                let s = self.tw[2 * b + 1];
                for k in 0..i {
                    let lo = base + k;
                    let hi = base + i + k;
                    let x1r = work[2 * lo];
                    let x1i = work[2 * lo + 1];
                    let x2r = work[2 * hi];
                    let x2i = work[2 * hi + 1];
                    let t1 = s * x2r + c * x2i;
                    let t2 = x2r * c - x2i * s;
                    let hi2r = (t1 + x1r) * 0.5f32;
                    work[2 * lo] = hi2r - t1;
                    work[2 * hi] = hi2r;
                    let hi2i = (x1i - t2) * 0.5f32;
                    work[2 * lo + 1] = t2 + hi2i;
                    work[2 * hi + 1] = hi2i;
                }
            }
            i >>= 1;
        }

        // ── untangle pass into the real half-spectrum: k = 1 … half/2
        //    (inclusive) pairs complex[bitrev[k]/2] with
        //    complex[bitrev[half-k]/2]; the last k pairs a bin with itself
        //    and its second store wins ────────────────────────────────────
        let bitrev = &self.bitrev;
        let tw = &self.tw;
        for k in 1..=half / 2 {
            let v = bitrev[k];
            let w = bitrev[half - k];
            let ts = tw[v];
            let tc = tw[v + 1];
            let vr = work[v];
            let vi = work[v + 1];
            let wr = work[w];
            let wi = work[w + 1];
            let d_im = vi - wi;
            let d_re = vr - wr;
            let sum_wi = (wi + wi) + d_im; // vi + wi
            let a = ts * d_re - tc * sum_wi;
            let b_ = tc * d_re + ts * sum_wi;
            let sum_wr = (wr + wr) + d_re; // vr + wr
            let v_new_r = (sum_wr + a) * 0.5f32;
            let v_new_i = (b_ + d_im) * 0.5f32;
            if v != w {
                work[v] = v_new_r;
                work[w] = v_new_r - a;
                work[v + 1] = v_new_i;
                work[w + 1] = v_new_i - d_im;
            } else {
                work[v] = v_new_r - a;
                work[v + 1] = v_new_i - d_im;
            }
        }

        // DC fold: data[0].re += data[0].im ; data[0].im = 0
        work[0] += work[1];
        work[1] = 0.0;

        // magnitudes read back at bit-reversed order (DoFFTToMagnitude)
        for k in 0..half {
            let f = bitrev[k];
            let re = work[f];
            let im = work[f + 1];
            out[k] = (re * re + im * im).sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive DFT magnitude in f64 of the *windowed* frame (the engine
    /// multiplies by its Hamming window before transforming).
    fn naive_dft_mag(fft: &SpecEngFft, x: &[f32]) -> Vec<f64> {
        let n = x.len();
        let wx: Vec<f64> = x
            .iter()
            .zip(fft.window.iter())
            .map(|(&v, &w)| v as f64 * w as f64)
            .collect();
        (0..n / 2)
            .map(|k| {
                let mut re = 0f64;
                let mut im = 0f64;
                for (i, &v) in wx.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                    re += v * a.cos();
                    im += v * a.sin();
                }
                (re * re + im * im).sqrt()
            })
            .collect()
    }

    #[allow(dead_code)]
    fn _unused(x: &[f32]) -> Vec<f64> {
        let n = x.len();
        (0..n / 2)
            .map(|k| {
                let mut re = 0f64;
                let mut im = 0f64;
                for (i, &v) in x.iter().enumerate() {
                    let a = -2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                    re += v as f64 * a.cos();
                    im += v as f64 * a.sin();
                }
                (re * re + im * im).sqrt()
            })
            .collect()
    }

    #[test]
    fn matches_naive_dft_up_to_the_originals_scaling() {
        let n = 256;
        let mut fft = SpecEngFft::new(n);
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / n as f32;
                (2.0 * std::f32::consts::PI * 5.0 * t).sin()
                    + 0.4 * (2.0 * std::f32::consts::PI * 23.0 * t).cos()
            })
            .collect();
        let mut out = vec![0f32; n / 2];
        fft.magnitudes(&input, &mut out);
        let want = naive_dft_mag(&fft, &input);
        for (k, (a, b)) in out.iter().zip(want.iter()).enumerate() {
            // out = want/128 up to f32 rounding
            let scaled = *a as f64 * 128.0;
            let tol = 1e-3 * (1.0 + b.abs());
            assert!(
                (scaled - b).abs() < tol,
                "bin {k}: got {scaled}, want {b}"
            );
        }
    }

    #[test]
    fn dc_bin_is_the_scaled_sample_sum() {
        let n = 256;
        let mut fft = SpecEngFft::new(n);
        let mut out = vec![0f32; n / 2];
        fft.magnitudes(&vec![0.5f32; n], &mut out);
        let want = naive_dft_mag(&fft, &vec![0.5f32; n])[0] / 128.0;
        assert!((out[0] as f64 - want).abs() < 1e-5, "dc = {}", out[0]);
    }

    #[test]
    fn work_buffer_is_reused_safely() {
        let n = 256;
        let mut fft = SpecEngFft::new(n);
        let mut out = vec![0f32; n / 2];
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();
        let mut first = vec![0f32; n / 2];
        fft.magnitudes(&a, &mut out);
        first.copy_from_slice(&out);
        fft.magnitudes(&b, &mut out);
        fft.magnitudes(&a, &mut out);
        assert_eq!(first, out);
    }
}
