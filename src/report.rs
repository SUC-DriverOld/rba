//! Output formatting: a result block for one file, a table for a directory.

use unicode_width::UnicodeWidthStr;

use crate::analyze::Analysis;

const HEADER_FILE: &str = "FILE";
const HEADER_BPM: &str = "BPM";
/// Gap between the two columns.
const COLUMN_GAP: &str = "  ";

/// Print one analysis result.
///
/// Names the file and labels the unit: a bare number on its own is easy to
/// misread, especially when the command was run a while ago. Detail goes to
/// stderr under `--verbose`.
pub fn print_single(a: &Analysis, verbose: bool) {
    if !a.name.is_empty() {
        println!("{}\tBPM: {:.2}", a.name, a.bpm);
    }
    else{
        println!("BPM: {:.2}", a.bpm);
    }
    if verbose {
        eprint!("{}", details(a));
    }
}

/// Human-readable breakdown, written to stderr.
pub fn details(a: &Analysis) -> String {
    let refinement = match a.trace.refined_bpm {
        Some(v) => format!("{v:.4} (fit {} onsets)", a.trace.refine_onsets),
        None => "skipped".to_string(),
    };
    format!(
        "  file            : {}\n\
         \x20 samples         : {} ({:.2} s @ {} Hz)\n\
         \x20 thumbnail elems : {}\n\
         \x20 peak-hold decay : {}\n\
         \x20 onsets (slope)  : {}\n\
         \x20 onsets (classic): {}\n\
         \x20 base BPM (0.1)  : {:.2}\n\
         \x20 after fold-up   : {:.2}\n\
         \x20 refined         : {}\n\
         \x20 short-clip rule : {}\n",
        a.name,
        a.sample_count,
        a.duration,
        a.sample_rate,
        a.trace.thumbnail_elements,
        a.trace.peak_hold_decay_div,
        a.trace.onset_count,
        a.trace.classic_onset_count,
        a.trace.base_bpm,
        a.trace.folded_bpm,
        refinement,
        if a.trace.short_clip_applied { "applied" } else { "n/a" },
    )
}

/// One row of the directory report.
#[derive(Debug)]
pub enum Row {
    Ok(Analysis),
    Failed { name: String, message: String },
}

impl Row {
    pub fn name(&self) -> &str {
        match self {
            Row::Ok(a) => &a.name,
            Row::Failed { name, .. } => name,
        }
    }
}

fn bpm_cell(row: &Row) -> String {
    match row {
        Row::Ok(a) => format!("{:.2}", a.bpm),
        Row::Failed { .. } => "FAILED".to_string(),
    }
}

/// Pad on the right to `width` **display columns** (not chars, so CJK names
/// stay aligned).
fn pad_right(s: &str, width: usize) -> String {
    format!("{s}{}", " ".repeat(width.saturating_sub(s.width())))
}

/// Pad on the left to `width` display columns.
fn pad_left(s: &str, width: usize) -> String {
    format!("{}{s}", " ".repeat(width.saturating_sub(s.width())))
}

/// Render the directory report: a plain `FILE / BPM` table, plus a short list of
/// reasons underneath if anything failed.
pub fn render_table(rows: &[Row]) -> String {
    let name_w = rows
        .iter()
        .map(|r| r.name().width())
        .chain(std::iter::once(HEADER_FILE.width()))
        .max()
        .unwrap_or(HEADER_FILE.len());
    let bpm_w = rows
        .iter()
        .map(|r| bpm_cell(r).width())
        .chain(std::iter::once(HEADER_BPM.width()))
        .max()
        .unwrap_or(HEADER_BPM.len());

    let mut out = String::new();
    out.push_str(&pad_right(HEADER_FILE, name_w));
    out.push_str(COLUMN_GAP);
    out.push_str(&pad_left(HEADER_BPM, bpm_w));
    out.push('\n');
    out.push_str(&"-".repeat(name_w));
    out.push_str(COLUMN_GAP);
    out.push_str(&"-".repeat(bpm_w));
    out.push('\n');

    for row in rows {
        out.push_str(&pad_right(row.name(), name_w));
        out.push_str(COLUMN_GAP);
        out.push_str(&pad_left(&bpm_cell(row), bpm_w));
        out.push('\n');
    }

    let failures: Vec<&Row> = rows
        .iter()
        .filter(|r| matches!(r, Row::Failed { .. }))
        .collect();
    if !failures.is_empty() {
        out.push('\n');
        out.push_str(&format!(
            "{} file{} could not be analyzed:\n",
            failures.len(),
            if failures.len() == 1 { "" } else { "s" }
        ));
        for f in failures {
            if let Row::Failed { name, message } = f {
                out.push_str(&format!("  {name} — {message}\n"));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::Trace;

    fn mk(name: &str, bpm: f64) -> Analysis {
        Analysis {
            name: name.to_string(),
            bpm,
            sample_count: 44100,
            sample_rate: 44100,
            duration: 1.0,
            trace: Trace::default(),
        }
    }

    #[test]
    fn table_has_header_and_rows() {
        let rows = vec![
            Row::Ok(mk("a.wav", 120.0)),
            Row::Ok(mk("bbbbbbbbbb.wav", 99.5)),
            Row::Failed {
                name: "broken.wav".into(),
                message: "decoder exploded".into(),
            },
        ];
        let t = render_table(&rows);
        assert!(t.contains("FILE"));
        assert!(t.contains("BPM"));
        assert!(t.contains("120.00"));
        assert!(t.contains("99.50"));
        // Failures appear in the table and are explained below it.
        assert!(t.contains("FAILED"));
        assert!(t.contains("could not be analyzed"));
        assert!(t.contains("decoder exploded"));
    }

    #[test]
    fn successful_run_has_no_failure_section() {
        let rows = vec![Row::Ok(mk("a.wav", 120.0))];
        let t = render_table(&rows);
        assert!(!t.contains("FAILED"));
        assert!(!t.contains("could not be analyzed"));
    }

    #[test]
    fn singular_failure_is_worded_correctly() {
        let rows = vec![Row::Failed {
            name: "x.wav".into(),
            message: "boom".into(),
        }];
        let t = render_table(&rows);
        assert!(t.contains("1 file could not be analyzed"), "{t}");
    }

    #[test]
    fn the_rule_matches_the_column_widths() {
        let rows = vec![Row::Ok(mk("a.wav", 120.0))];
        let rendered = render_table(&rows);
        let lines: Vec<&str> = rendered.lines().collect();
        // Header, rule, one row.
        assert_eq!(lines.len(), 3);
        let (header, rule, row) = (lines[0], lines[1], lines[2]);
        assert_eq!(header.len(), rule.len(), "rule must span the header");
        assert_eq!(header.len(), row.len(), "row must line up with the header");
        assert!(rule.starts_with("---"), "rule: {rule}");
        assert!(rule.contains("  "), "rule: {rule}");
        assert!(!rule.contains('+'), "plain dashes only: {rule}");
    }

    #[test]
    fn columns_align_across_rows() {
        let rows = vec![
            Row::Ok(mk("short.wav", 90.0)),
            Row::Ok(mk("a_much_longer_name.wav", 120.25)),
        ];
        let rendered = render_table(&rows);
        let lines: Vec<&str> = rendered.lines().collect();
        // BPM is the last column and right-aligned, so every value *ends* in
        // the same column as the header's.
        let bpm_end = lines[0].find("BPM").expect("header has BPM") + "BPM".len();
        for line in &lines[2..] {
            assert_eq!(line.len(), bpm_end, "line: {line}");
        }
    }
}
