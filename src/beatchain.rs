//! The beat chain: the original's final BPM stage.
//!
//! Ports `CPfBeatBuilder::ClusterFilterPoints` and `BuildBeatPointsAlt`
//! (re/17). The chain stage takes the divisor-8 peak-hold onsets, links them
//! into interval-consistent chains, and returns the winning chain's mean
//! inter-onset interval `ioI`; the BPM is `30 · sample_rate / ioI`.
//!
//! Everything here is `f32`, like the DLL. Integer conversions follow the
//! original's saturating `(u32)double` and truncating `cvttss2si` semantics.

use crate::impact::ImpactPoints;
use crate::thumbnail::{sam_per_meas, Thumbnail};

/// Phase 10: chains whose first/last index difference is ≤ this are dropped.
pub const CHAIN_MIN_INDEX_SPAN: i64 = 31;

/// §3.1-A — the chain stage's saturating `(u32)double` conversion.
#[inline]
pub fn u32_sat(x: f64) -> u32 {
    let (m, hi) = if x >= 2_147_483_648.0 {
        (2_147_483_648.0, 0x8000_0000u32)
    } else {
        (0.0, 0)
    };
    let t = x.clamp(0.0, 4_294_967_295.0) - m;
    (t as u32) ^ hi
}

/// One scored chain, for `--verbose` diagnostics.
#[derive(Debug, Clone, Default)]
pub struct ChainDetail {
    pub weight: f32,
    pub members: usize,
    pub span: i64,
    pub mean_conf: f32,
    /// What ioI this chain would report if it won.
    pub own_io_i: f32,
}

/// Diagnostics from the chain stage (for `--verbose`).
#[derive(Debug, Clone, Default)]
pub struct ChainTrace {
    /// `SamPerMeas(bpm) >> 3`, as f32.
    pub unit0: f32,
    /// Spacing re-estimate before clustering.
    pub interval_est: f32,
    /// Connections with quality > 0.
    pub candidates: usize,
    /// Connections consumed before the coverage break.
    pub processed: usize,
    /// Coverage when the assembly stopped.
    pub coverage: f64,
    /// Chains in the map when assembly stopped.
    pub chains_total: usize,
    /// Chains surviving phases 9-10.
    pub chains_kept: usize,
    /// Members of the winning chain.
    pub best_members: usize,
    /// Final interval (samples per eighth).
    pub io_i: f32,
    /// Per scored chain: `(weight, member count, span, mean_conf, own ioI)`.
    pub chain_details: Vec<ChainDetail>,
}

/// `CalcPointConfidence` (re/17 §3.2) — 16-element rise around the onset.
///
/// `Σ thumb[idx..idx+7] − Σ thumb[idx-8..idx-1]` with every index clamped to
/// `[0, count-1]`; the sums accumulate sequentially in f32 (each window is at
/// most 8 elements).
pub fn calc_point_confidence(thumb: &[f32], pos: i64) -> f32 {
    if thumb.is_empty() {
        return 0.0;
    }
    let n1 = thumb.len() as i64 - 1;
    let idx = pos / 256; // __divdi3: signed, truncating

    let clamp = |v: i64| v.clamp(0, n1);
    let lo1 = clamp(idx - 8);
    let lo2 = clamp(idx - 1);
    let hi1 = clamp(idx);
    let hi2 = clamp(idx + 7);

    let mut s_lo = 0f32;
    if lo2 >= lo1 {
        for i in lo1..=lo2 {
            s_lo += thumb[i as usize];
        }
    }
    let mut s_hi = 0f32;
    if hi2 >= hi1 {
        for i in hi1..=hi2 {
            s_hi += thumb[i as usize];
        }
    }
    0.0f32 - s_lo + s_hi
}

/// `ProximityConfFilter` (re/17 §3.6) — non-maximum suppression.
///
/// Within ±`trunc(0.45·I)` samples only the most confident candidate
/// survives; survivors keep their relative order. Returns kept indices.
pub fn proximity_conf_filter(pos: &[i64], conf: &[f32], interval: f32) -> Vec<usize> {
    let n = pos.len();
    let limit = (0.45f32 * interval) as i64;
    let mut nxt: Vec<usize> = (1..=n).collect(); // nxt[n-1] == n doubles as the end sentinel
    let mut prv: Vec<i64> = (-1..n as i64).collect();
    let mut head: i64 = 0;

    let mut cur = head;
    while (cur as usize) < n {
        let nxt_cur = nxt[cur as usize]; // advance before a possible erase
        let mut best = cur as usize;
        let mut i = cur; // backward scan (includes cur)
        while i >= 0 && limit > pos[cur as usize] - pos[i as usize] {
            if conf[i as usize] > conf[best] {
                best = i as usize;
            }
            i = prv[i as usize];
        }
        let mut j = cur as usize; // forward scan (includes cur)
        while j < n && limit > pos[j] - pos[cur as usize] {
            if conf[j] > conf[best] {
                best = j;
            }
            j = nxt[j];
        }
        if best != cur as usize && conf[best] > conf[cur as usize] {
            let p = prv[cur as usize];
            let s = nxt[cur as usize];
            if p >= 0 {
                nxt[p as usize] = s;
            } else {
                head = s as i64;
            }
            if s < n {
                prv[s] = p;
            }
        }
        cur = nxt_cur as i64;
    }

    let mut out = Vec::new();
    let mut i = head;
    while (i as usize) < n {
        out.push(i as usize);
        i = nxt[i as usize] as i64;
    }
    out
}

/// One candidate connection (phase 6): points `a → b` with quality > 0.
#[derive(Clone, Copy)]
struct Candidate {
    a: usize,
    b: usize,
    q: f32,
}

/// Phase 6 — all pairs `(i, j)`, `i < j`, within 32 units, scored by
/// `CalcConQuality` in f32 like the original.
fn connection_candidates(pos: &[i64], conf: &[f32], interval: f32) -> Vec<Candidate> {
    let n = pos.len();
    let mut out = Vec::new();
    // Negated comparison on purpose: it rejects NaN like the original's C
    // comparison would.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if n < 2 || !(interval > 0.0) {
        return out;
    }

    // Break condition: u32_sat(d/I + 0.5) > 32  ⇔  d ≥ 32.5·I (d ≥ 0).
    let reach = (32.5f32 * interval) as f64;
    let interval_f = interval as f64;

    for i in 0..n - 1 {
        let lim = pos[i] as f64 + reach;
        let mut j = i + 1;
        while j < n && (pos[j] as f64) < lim {
            let d = pos[j] - pos[i];
            let n_units = ((d as f64 / interval_f + 0.5).trunc()) as i64;

            let x = d as f32 / interval;
            let r = (x + 0.5f32) as i32; // cvttss2si: truncating
            let frac = x - r as f32;
            let v = 1.0f32 - frac.abs();
            let phase = if v > 0.8f32 {
                if v <= 0.98f32 {
                    (v - 0.8f32) / 0.18f32
                } else {
                    1.0
                }
            } else {
                0.0
            };
            let tol = if n_units > 8 {
                (32 - n_units) as f32 / 24.0f32
            } else {
                1.0
            };
            let q = (phase * tol * conf[i]) * conf[j];

            if n_units >= 1 && q > 0.0 {
                out.push(Candidate { a: i, b: j, q });
            }
            j += 1;
        }
    }

    // Sorted by quality descending, stable (like std::sort with `q > q'`).
    out.sort_by(|x, y| y.q.partial_cmp(&x.q).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// The BuildBeatPointsAlt / phase-5 spacing fit (re/17 §4.2).
///
/// Consecutive spacings that are 1-3 units of `unit0` within ±5 % vote; the
/// mean of `d/n` wins, else `unit0`.
pub fn reestimate_interval(points: &[i64], unit0: f32) -> f32 {
    let lo = (unit0 * -0.05f32 + unit0) as i64;
    let hi = (unit0 * 0.05f32 + unit0) as i64;
    let mut total: i64 = 0;
    let mut cnt = 0;
    let mut prev: Option<i64> = None;
    for &cur in points {
        if let Some(p) = prev {
            let d = cur - p;
            let m = u32_sat(d as f64 / unit0 as f64 + 0.5);
            if (1..=3).contains(&m) {
                let q = d / (m as i64);
                if (lo..=hi).contains(&q) {
                    total += q;
                    cnt += 1;
                }
            }
        }
        prev = Some(cur);
    }
    if cnt != 0 {
        (total as f32) / (cnt as f32)
    } else {
        unit0
    }
}

/// Assembly state for `cluster_filter_points`.
struct Assembly {
    n: usize,
    index: Vec<i64>,
    chain_id: Vec<i32>,
    active: Vec<bool>,
    nxt: Vec<Vec<usize>>,
    prv: Vec<Vec<usize>>,
    chains: std::collections::BTreeMap<u32, Vec<usize>>,
    next_id: u32,
    active_count: usize,
    assigned_count: usize,
}

impl Assembly {
    /// `DeactivateAllNext/Prev` — clears `active` on every descendant.
    ///
    /// The DLL recurses through *already inactive* nodes as well (no
    /// active-flag check), so traversal must not stop at inactive points;
    /// clearing is idempotent, and the graphs are DAGs (links always point
    /// towards strictly greater/lesser indices), so this terminates.
    fn deactivate_reach(&mut self, start: usize, forward: bool) {
        let mut stack: Vec<usize> = if forward {
            self.nxt[start].clone()
        } else {
            self.prv[start].clone()
        };
        while let Some(p) = stack.pop() {
            if self.active[p] {
                self.active[p] = false;
                self.active_count -= 1;
                if self.chain_id[p] != -1 {
                    self.assigned_count -= 1;
                }
            }
            let more = if forward {
                &self.nxt[p]
            } else {
                &self.prv[p]
            };
            stack.extend_from_slice(more);
        }
    }

    /// `GetEarliestLink` — min index over a and its prev-descendants.
    fn earliest(&self, a: usize) -> i64 {
        let mut best = self.index[a];
        let mut stack = self.prv[a].clone();
        while let Some(p) = stack.pop() {
            if self.index[p] < best {
                best = self.index[p];
            }
            stack.extend_from_slice(&self.prv[p]);
        }
        best
    }

    /// `GetLatestLink` — max index over a and its next-descendants.
    fn latest(&self, a: usize) -> i64 {
        let mut best = self.index[a];
        let mut stack = self.nxt[a].clone();
        while let Some(p) = stack.pop() {
            if self.index[p] > best {
                best = self.index[p];
            }
            stack.extend_from_slice(&self.nxt[p]);
        }
        best
    }

    /// `CPfBeatPoint::Disconnect` — erase `other` from both link lists.
    fn disconnect(&mut self, this: usize, other: usize) {
        self.nxt[this].retain(|&x| x != other);
        self.prv[this].retain(|&x| x != other);
    }

    fn disconnect_all_next(&mut self, a: usize) {
        let links = self.nxt[a].clone();
        for p in links {
            self.disconnect(p, a);
        }
        self.nxt[a].clear();
    }

    fn disconnect_all_prev(&mut self, a: usize) {
        let links = self.prv[a].clone();
        for p in links {
            self.disconnect(p, a);
        }
        self.prv[a].clear();
    }

    /// Rebuild a chain's member list from the (possibly re-labelled) points.
    fn rebuild_chain(&mut self, cid: u32) {
        let Some(ch) = self.chains.get_mut(&cid) else {
            return;
        };
        ch.clear();
        for p in 0..self.n {
            if self.chain_id[p] == cid as i32 && self.active[p] {
                ch.push(p);
            }
        }
        // Members are recorded as array positions; index == position here.
        ch.sort_unstable();
    }

    fn drop_point(&mut self, p: usize) {
        if self.active[p] {
            self.active[p] = false;
            self.active_count -= 1;
            if self.chain_id[p] != -1 {
                self.assigned_count -= 1;
            }
        }
    }
}

/// `CPfBeatBuilder::ClusterFilterPoints` (re/17 §2) — faithful port.
///
/// Returns `(ioI, emitted positions)`; the emitted positions are the union of
/// surviving chains' onsets (the DLL feeds them to `InterpolatePoints` /
/// `ExtrapolatePoints`, which do not influence the BPM).
pub fn cluster_filter_points(
    thumb: &[f32],
    i0: f32,
    points: &[i64],
    trace: &mut ChainTrace,
) -> (f32, Vec<i64>) {
    // ── phase 1: points + confidence ────────────────────────────────────
    let mut pos: Vec<i64> = Vec::with_capacity(points.len());
    let mut conf: Vec<f32> = Vec::with_capacity(points.len());
    let mut min_c = 0f32;
    let mut max_c = 0f32;
    for (k, &p) in points.iter().enumerate() {
        let c = calc_point_confidence(thumb, p);
        pos.push(p);
        conf.push(c);
        if k == 0 {
            min_c = c;
            max_c = c;
        } else {
            min_c = min_c.min(c);
            max_c = max_c.max(c);
        }
    }

    // ── phase 2: normalise confidences to [0, 1] ────────────────────────
    let rng = max_c - min_c;
    let scale = if rng != 0.0 { rng } else { 1.0 };
    for c in conf.iter_mut() {
        *c = (*c - min_c) / scale;
    }

    // ── phase 3: proximity thinning (uses the *input* interval) ─────────
    let kept = proximity_conf_filter(&pos, &conf, i0);

    // ── phase 4: re-index (index == array position after thinning) ──────
    pos = kept.iter().map(|&i| pos[i]).collect();
    conf = kept.iter().map(|&i| conf[i]).collect();
    let n = pos.len();

    // ── phase 5: re-estimate the interval from 1-3-unit spacings ────────
    let interval = reestimate_interval(&pos, i0);

    // ── phase 6: candidate connections, sorted by quality (desc) ────────
    let candidates = connection_candidates(&pos, &conf, interval);
    trace.candidates = candidates.len();

    // ── phase 7: greedy chain assembly ──────────────────────────────────
    let mut asm = Assembly {
        n,
        index: (0..n as i64).collect(),
        chain_id: vec![-1; n],
        active: vec![true; n],
        nxt: vec![Vec::new(); n],
        prv: vec![Vec::new(); n],
        chains: Default::default(),
        next_id: 1,
        active_count: n,
        assigned_count: 0,
    };

    let mut processed = 0usize;
    for cand in &candidates {
        let (a, b) = (cand.a, cand.b);
        processed += 1;
        if !(asm.active[a] && asm.active[b]) {
            continue; // skips the coverage check, like the DLL
        }
        let (ca, cb) = (asm.chain_id[a], asm.chain_id[b]);
        let mut connect = false;
        if ca == -1 {
            if cb == -1 {
                // both unassigned → new chain
                asm.chain_id[a] = asm.next_id as i32;
                asm.chain_id[b] = asm.next_id as i32;
                asm.chains.insert(asm.next_id, vec![a, b]);
                asm.next_id += 1;
                asm.assigned_count += 2;
                connect = true;
            } else if asm.index[a] < asm.earliest(b) {
                // A prepends itself to B's chain
                asm.deactivate_reach(b, false);
                asm.disconnect_all_prev(b);
                asm.chain_id[a] = cb;
                asm.assigned_count += 1;
                let cbu = cb as u32;
                asm.rebuild_chain(cbu);
                connect = true;
            }
        } else if cb == -1 {
            if asm.latest(a) < asm.index[b] {
                // B appends itself to A's chain
                asm.deactivate_reach(a, true);
                asm.disconnect_all_next(a);
                asm.chain_id[b] = ca;
                asm.assigned_count += 1;
                let cau = ca as u32;
                asm.rebuild_chain(cau);
                connect = true;
            }
        } else if ca != cb {
            let ea = asm.earliest(a);
            let la = asm.latest(a);
            let eb = asm.earliest(b);
            let lb = asm.latest(b);
            if ea <= eb && la >= lb {
                // B's span inside A's — B's chain is dropped entirely.
                asm.deactivate_reach(b, false);
                asm.disconnect_all_prev(b);
                asm.deactivate_reach(b, true);
                asm.disconnect_all_next(b);
                asm.drop_point(b);
                asm.chains.remove(&(cb as u32));
            } else if ea >= eb && la <= lb {
                asm.deactivate_reach(a, false);
                asm.disconnect_all_prev(a);
                asm.deactivate_reach(a, true);
                asm.disconnect_all_next(a);
                asm.drop_point(a);
                asm.chains.remove(&(ca as u32));
            } else if ea < eb {
                // merge B's chain into A's
                asm.deactivate_reach(a, true);
                asm.disconnect_all_next(a);
                asm.deactivate_reach(b, false);
                asm.disconnect_all_prev(b);
                let keep_id = ca as u32;
                for p in 0..n {
                    if asm.chain_id[p] == cb {
                        asm.chain_id[p] = keep_id as i32;
                    }
                }
                asm.chains.remove(&(cb as u32));
                asm.rebuild_chain(keep_id);
                connect = true;
            }
        }

        if connect {
            // Connect(A, B); Connect(B, A) — A is always the earlier point.
            asm.nxt[a].push(b); // index[b] > index[a]
            asm.prv[b].push(a);
        }

        if asm.active_count != 0 {
            let coverage = asm.assigned_count as f64 / asm.active_count as f64;
            if coverage >= 0.8 {
                break;
            }
        }
    }

    trace.processed = processed;
    trace.coverage = if asm.active_count != 0 {
        asm.assigned_count as f64 / asm.active_count as f64
    } else {
        0.0
    };
    trace.chains_total = asm.chains.len();

    // ── phase 8: chain scoring ──────────────────────────────────────────
    let mut spans: Vec<i64> = Vec::new();
    let mut items_list: Vec<Vec<usize>> = Vec::new();
    let mut means: Vec<f32> = Vec::new();
    let mut max_span: i64 = -1;
    let mut min_mc = 0f32;
    let mut max_mc = 0f32;
    for (&cid, items) in asm.chains.iter() {
        let _ = cid;
        let span = pos[*items.last().unwrap()] - pos[*items.first().unwrap()];
        let mut s = 0f32;
        for &ix in items {
            s += conf[ix];
        }
        let mean_c = if items.is_empty() { 0.0 } else { s / items.len() as f32 };
        spans.push(span);
        items_list.push(items.clone());
        means.push(mean_c);
        if max_span == -1 {
            max_span = span;
            min_mc = mean_c;
            max_mc = mean_c;
        } else {
            max_span = max_span.max(span);
            max_mc = max_mc.max(mean_c);
            min_mc = min_mc.min(mean_c);
        }
    }
    if max_span == 0 {
        max_span = 1;
    }
    let span_f = max_span as f32;
    let mut conf_range = max_mc - min_mc;
    if conf_range == 0.0 {
        conf_range = 1.0;
    }

    let mut scored: Vec<(f32, usize)> = Vec::new(); // (weight, slot into items_list)
    for (slot, (&span, &mc)) in spans.iter().zip(means.iter()).enumerate() {
        let weight = (span as f32 / span_f) * ((mc - min_mc) / conf_range);
        // Diagnostics: this chain's own ioI (what it would report if it won).
        let mut sd: i64 = 0;
        let mut sn: u32 = 0;
        let mut prev: Option<i64> = None;
        for &ix in &items_list[slot] {
            let cur = pos[ix];
            if let Some(p) = prev {
                let d = cur - p;
                sd += d;
                sn = sn.wrapping_add(u32_sat(d as f64 / interval as f64 + 0.5));
            }
            prev = Some(cur);
        }
        let own = if sn != 0 {
            (sd as f32) / (sn as f32)
        } else {
            interval
        };
        trace.chain_details.push(ChainDetail {
            weight,
            members: items_list[slot].len(),
            span,
            mean_conf: mc,
            own_io_i: own,
        });
        scored.push((weight, slot));
    }
    // Stable descending sort, like `std::list::sort` with `q > q'`.
    scored.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));

    // ── phase 9: overlap suppression ────────────────────────────────────
    let mut kept_chains: Vec<Vec<usize>> = Vec::new();
    for &(_, slot) in &scored {
        let items = &items_list[slot];
        let first = pos[*items.first().unwrap()];
        let last = pos[*items.last().unwrap()];
        let mut dropped = false;
        for kitems in &kept_chains {
            let kfirst = pos[*kitems.first().unwrap()];
            let klast = pos[*kitems.last().unwrap()];
            if !(klast < first || last < kfirst) {
                dropped = true;
                break;
            }
        }
        if !dropped {
            kept_chains.push(items.clone());
        }
    }

    // ── phase 10: reject short chains (first/last *index* span ≤ 31; the
    //     span is measured in thinned-array positions, not samples) ──────
    kept_chains.retain(|items| {
        let first = *items.first().unwrap() as i64;
        let last = *items.last().unwrap() as i64;
        last - first > CHAIN_MIN_INDEX_SPAN
    });

    trace.chains_kept = kept_chains.len();

    // ── phase 11: emit + final interval from the best chain ─────────────
    let mut emitted: Vec<i64> = Vec::new();
    for items in &kept_chains {
        emitted.extend(items.iter().map(|&ix| pos[ix]));
    }

    let mut io_i = interval;
    if let Some(items) = kept_chains.first() {
        let mut sum_d: i64 = 0;
        let mut sum_n: u32 = 0;
        let mut prev: Option<i64> = None;
        for &ix in items {
            let cur = pos[ix];
            if let Some(p) = prev {
                let d = cur - p;
                sum_d += d;
                sum_n = sum_n.wrapping_add(u32_sat(d as f64 / interval as f64 + 0.5));
            }
            prev = Some(cur);
        }
        trace.best_members = items.len();
        if sum_n != 0 {
            io_i = (sum_d as f32) / (sum_n as f32);
        }
    }
    trace.io_i = io_i;
    (io_i, emitted)
}

/// `CPfBeatBuilder::BuildBeatPointsAlt` (re/17 §4.2).
///
/// `onsets8` is `BuildImpactPoints(peakThumb, 8)` — the DLL re-detects
/// onsets with divisor 8 for the chain stage. The refined interval comes
/// back from `cluster_filter_points` and the BPM is
/// `60 / (8·ioI·0.25 / sr) = 30·sr / ioI`.
pub fn beat_chain_bpm(
    spec_thumb: &Thumbnail,
    onsets8: &ImpactPoints,
    bpm_prior: f32,
    sample_rate: u32,
    trace: &mut ChainTrace,
) -> f32 {
    let unit0 = (sam_per_meas(bpm_prior, sample_rate) >> 3) as f32;
    trace.unit0 = unit0;

    let interval = reestimate_interval(&onsets8.points, unit0);
    trace.interval_est = interval;

    let (io_i, _emitted) =
        cluster_filter_points(&spec_thumb.data, interval, &onsets8.points, trace);

    if sample_rate != 0 {
        let p = (8.0f32 * io_i * 0.25f32) / sample_rate as f32;
        if p != 0.0 {
            // The reference implementation divides in f64 and stores to f32
            // (Python float semantics); keep the same rounding here.
            return (60.0f64 / p as f64) as f32;
        }
    }
    bpm_prior
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u32_sat_clamps_and_saturates() {
        assert_eq!(u32_sat(0.0), 0);
        assert_eq!(u32_sat(2.7), 2);
        assert_eq!(u32_sat(-3.0), 0);
        assert_eq!(u32_sat(1e18), 0xFFFF_FFFF);
    }

    #[test]
    fn confidence_uses_a_16_element_window() {
        let thumb: Vec<f32> = (0..64).map(|i| i as f32).collect();
        // pos 10·256 → idx 10: sums thumb[10..=17] - thumb[2..=9]
        let want: f32 = (10..=17i32).map(|i| i as f32).sum::<f32>()
            - (2..=9i32).map(|i| i as f32).sum::<f32>();
        assert!((calc_point_confidence(&thumb, 10 * 256) - want).abs() < 1e-3);
        // Out-of-range index clamps: idx -100 → window clamps to [0, ...]
        let c = calc_point_confidence(&thumb, -100 * 256);
        assert!(c.is_finite());
    }

    #[test]
    fn proximity_filter_keeps_the_strongest_in_each_neighbourhood() {
        // Two clusters 1000 samples apart; within each, the highest conf wins.
        let pos = vec![100, 200, 300, 1100, 1200, 1300];
        let conf = vec![0.1, 0.9, 0.2, 0.3, 0.8, 0.4];
        let kept = proximity_conf_filter(&pos, &conf, 500.0); // limit 225
        assert_eq!(kept, vec![1, 4]);
    }

    #[test]
    fn reestimate_prefers_voted_mean() {
        // unit0 1000; spacings 1000, 2000, 3000 → votes 1000, 1000, 1000
        let pts = vec![0, 1000, 3000, 6000];
        let v = reestimate_interval(&pts, 1000.0);
        assert!((v - 1000.0).abs() < 1e-3, "got {v}");
    }

    #[test]
    fn reestimate_falls_back_to_unit0() {
        let pts = vec![0, 7, 50_000];
        assert_eq!(reestimate_interval(&pts, 1000.0), 1000.0);
    }

    #[test]
    fn empty_input_is_safe() {
        let mut trace = ChainTrace::default();
        let (io, emitted) = cluster_filter_points(&[], 1000.0, &[], &mut trace);
        assert_eq!(io, 1000.0);
        assert!(emitted.is_empty());
        assert_eq!(trace.chains_kept, 0);
    }

    fn perfect_grid_thumb(n: usize) -> Vec<f32> {
        // Rising ramp so every onset has the same confidence shape.
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn perfect_grid_recovers_the_interval() {
        // Onsets every 1000 samples over 100 s; unit0 = 1000.
        let n_elem = 44_100 * 100 / 256;
        let thumb = perfect_grid_thumb(n_elem);
        let mut points = Vec::new();
        let mut p = 0i64;
        while p < n_elem as i64 * 256 {
            points.push(p);
            p += 1000;
        }
        let mut trace = ChainTrace::default();
        let (io_i, _emitted) = cluster_filter_points(&thumb, 1000.0, &points, &mut trace);
        assert!((io_i - 1000.0).abs() < 1.0, "ioI = {io_i}");
        assert!(trace.chains_kept >= 1);
        assert!(trace.best_members > 10);
    }
}
