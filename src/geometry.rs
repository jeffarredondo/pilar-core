use nalgebra::DMatrix;

// ── Poincaré Ball Distance (N-D) ──────────────────────────────────────────────

/// Computes the hyperbolic distance between two points on the Poincaré ball.
/// Works for any dimension — disk (2D), ball (3D), or higher. Both points
/// must satisfy |p| < 1.
pub fn poincare_distance(u: &[f64], v: &[f64]) -> f64 {
    debug_assert_eq!(u.len(), v.len(), "points must have the same dimension");

    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum();

    debug_assert!(norm_u_sq < 1.0, "u is outside the Poincaré ball");
    debug_assert!(norm_v_sq < 1.0, "v is outside the Poincaré ball");

    let norm_diff_sq: f64 = u
        .iter()
        .zip(v.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();

    let denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq);
    let arg = 1.0 + 2.0 * norm_diff_sq / denom;
    // Only guard against floating-point rounding pushing arg slightly below
    // 1.0 (acosh's domain boundary) — do NOT add epsilon padding here. The
    // original Python clamped to 1.0 + 1e-8 unconditionally, which corrupts
    // distance(x, x): arg == 1.0 exactly for identical points, and acosh's
    // slope near 1.0 is steep (acosh(1+ε) ≈ √(2ε)), so that padding turned
    // a true distance of 0 into ~1.4e-4. Caught by test_poincare_origin.
    arg.max(1.0).acosh()
}

// ── Distance Matrix ───────────────────────────────────────────────────────────

/// Builds a pairwise Euclidean distance matrix from a set of feature vectors.
/// This operates in embedding space, before any geometry is assigned —
/// it's the input to classification, not the manifold itself.
pub fn distance_matrix(vectors: &[Vec<f64>]) -> DMatrix<f64> {
    let n = vectors.len();
    let mut d = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean(&vectors[i], &vectors[j]);
            d[(i, j)] = dist;
            d[(j, i)] = dist;
        }
    }
    d
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ── Gromov Delta ──────────────────────────────────────────────────────────────

/// Approximates Gromov delta hyperbolicity from a distance matrix.
/// Low delta (~0) = tree-like = hyperbolic.
/// High delta = spherical or flat.
/// O(n⁴) — only call this on small neighborhoods (k=5 or so), same as Python.
pub fn gromov_delta(d: &DMatrix<f64>) -> f64 {
    let n = d.nrows();
    let mut max_delta = 0.0_f64;

    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    let mut sums = [
                        d[(i, j)] + d[(k, l)],
                        d[(i, k)] + d[(j, l)],
                        d[(i, l)] + d[(j, k)],
                    ];
                    sums.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    let delta = (sums[0] - sums[1]) / 2.0;
                    max_delta = max_delta.max(delta);
                }
            }
        }
    }
    max_delta
}

// ── Eigenvalue Signature ──────────────────────────────────────────────────────

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum GeometryClass {
    Hyperbolic,
    Spherical,
    Flat,
}

/// The full output of neighborhood geometry classification — the decision
/// AND the evidence behind it. Earlier versions of this function returned
/// only `GeometryClass`, discarding the numbers that actually drove the
/// decision. Kept now because "how confidently did this land here" turns
/// out to matter as much as "where did it land" — see GeometryConfidence
/// on the Concept struct.
#[derive(Debug, Clone, Copy)]
pub struct EigenSignature {
    pub class: GeometryClass,
    /// Ratio of the 2nd to 1st normalized eigenvalue. High = spherical signal.
    pub eigenvalue_ratio: f64,
    /// How much the 1st eigenvalue dominates. High = flat signal.
    pub first_dominance: f64,
    /// Fraction of eigenvalues that are significantly negative. High = hyperbolic signal.
    pub neg_eigenvalue_fraction: f64,
}

/// Classifies local neighborhood geometry via eigenvalue analysis of the
/// double-centered distance matrix (standard MDS preprocessing).
/// Same three-way heuristic as before: dominant-pair ratio catches
/// spherical, negative-eigenvalue count catches hyperbolic, single
/// dominant eigenvalue catches flat — now returned alongside the raw
/// signal strengths instead of collapsing them into just the label.
pub fn eigenvalue_signature(d: &DMatrix<f64>) -> EigenSignature {
    let n = d.nrows();
    let ones = DMatrix::from_element(n, n, 1.0 / n as f64);
    let h = DMatrix::identity(n, n) - ones;
    let d_sq = d.map(|x| x * x);
    let b = -0.5 * &h * d_sq * &h;

    // Symmetrize to guard against floating point drift before eigendecomposition
    let sym = DMatrix::from_fn(n, n, |i, j| (b[(i, j)] + b[(j, i)]) / 2.0);
    let eigen = nalgebra::linalg::SymmetricEigen::new(sym);
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending

    let total: f64 = eigenvalues.iter().map(|x| x.abs()).sum();
    if total < 1e-10 {
        // Degenerate neighborhood — essentially no structure to read a
        // signal from at all (near-coincident points). Not "confidently
        // flat," just no evidence either way.
        return EigenSignature {
            class: GeometryClass::Flat,
            eigenvalue_ratio: 0.0,
            first_dominance: 0.0,
            neg_eigenvalue_fraction: 0.0,
        };
    }

    let normed: Vec<f64> = eigenvalues.iter().map(|x| x / total).collect();
    let first_dominance = normed[0];
    let ratio = if normed.len() > 1 {
        normed[1] / (normed[0] + 1e-10)
    } else {
        0.0
    };
    let neg_count = eigenvalues
        .iter()
        .filter(|&&x| x < -0.05 * eigenvalues[0].abs())
        .count();
    let neg_eigenvalue_fraction = neg_count as f64 / n as f64;

    let class = if ratio > 0.6 {
        GeometryClass::Spherical
    } else if neg_count > n / 3 {
        GeometryClass::Hyperbolic
    } else if first_dominance > 0.7 {
        GeometryClass::Flat
    } else {
        GeometryClass::Hyperbolic
    };

    EigenSignature {
        class,
        eigenvalue_ratio: ratio,
        first_dominance,
        neg_eigenvalue_fraction,
    }
}

// ── Möbius (Gyrovector) Addition ──────────────────────────────────────────────

/// Möbius addition on the Poincaré ball — the hyperbolic analog of vector
/// addition. `a ⊕ b` is "b, as seen from a frame where a is the new origin."
///
/// This is what makes recentering possible: a concept near the boundary
/// (r → 1) has `1 - r²` → 0 in the poincare_distance denominator, which is
/// exactly where f64 precision falls apart. Translating a whole periphery
/// neighborhood so its anchor point becomes the local origin moves every
/// point back into the well-conditioned region near r = 0, without
/// distorting any pairwise distance — it's an isometry.
///
/// Reference: Ungar's gyrovector formalism (same one "Hyperbolic Neural
/// Networks", Ganea et al. 2018, builds its layers on).
pub fn mobius_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());

    let dot_ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a_sq: f64 = a.iter().map(|x| x * x).sum();
    let norm_b_sq: f64 = b.iter().map(|x| x * x).sum();

    let coeff_a = 1.0 + 2.0 * dot_ab + norm_b_sq;
    let coeff_b = 1.0 - norm_a_sq;
    let denom = 1.0 + 2.0 * dot_ab + norm_a_sq * norm_b_sq;

    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (coeff_a * ai + coeff_b * bi) / denom)
        .collect()
}

/// Translates `x` into the frame where `origin` becomes the new center,
/// i.e. computes `(-origin) ⊕ x`. Distances between translated points are
/// identical to distances between the originals (isometry) — only the
/// coordinate representation changes, which is the whole point: same
/// geometry, better-conditioned numbers.
///
/// `translate_to_origin(origin, origin)` always lands exactly at the
/// zero vector — that's the property a recentering operation has to have.
pub fn translate_to_origin(origin: &[f64], x: &[f64]) -> Vec<f64> {
    let neg_origin: Vec<f64> = origin.iter().map(|v| -v).collect();
    mobius_add(&neg_origin, x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_origin() {
        assert!((poincare_distance(&[0.0, 0.0], &[0.0, 0.0])).abs() < 1e-10);
    }

    #[test]
    fn test_poincare_symmetry() {
        let u = [0.3, 0.1];
        let v = [0.5, 0.4];
        let d1 = poincare_distance(&u, &v);
        let d2 = poincare_distance(&v, &u);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_poincare_boundary_expansion() {
        let d_center = poincare_distance(&[0.1, 0.0], &[0.2, 0.0]);
        let d_boundary = poincare_distance(&[0.8, 0.0], &[0.9, 0.0]);
        assert!(d_boundary > d_center * 3.0);
    }

    #[test]
    fn test_poincare_3d() {
        // Same shape of test as 2D, just confirming N-D works at all —
        // H³ is the whole point of this generalization.
        let u = [0.1, 0.2, 0.0];
        let v = [0.3, 0.1, 0.2];
        let d = poincare_distance(&u, &v);
        assert!(d > 0.0);
        assert!((d - poincare_distance(&v, &u)).abs() < 1e-10);
    }

    #[test]
    fn test_translate_self_to_zero() {
        // A point translated into its own frame must land exactly at origin —
        // this is the defining property of "recenter on this point."
        let a = [0.6, 0.2, 0.1];
        let result = translate_to_origin(&a, &a);
        for coord in result {
            assert!(coord.abs() < 1e-10);
        }
    }

    #[test]
    fn test_translate_preserves_distance() {
        // The whole reason this is useful: recentering must not change
        // the hyperbolic distance between points, only their coordinates.
        let anchor = [0.7, 0.1, 0.0];
        let u = [0.75, 0.12, 0.05];
        let v = [0.6, 0.3, 0.1];

        let d_before = poincare_distance(&u, &v);

        let u_t = translate_to_origin(&anchor, &u);
        let v_t = translate_to_origin(&anchor, &v);
        let d_after = poincare_distance(&u_t, &v_t);

        assert!(
            (d_before - d_after).abs() < 1e-8,
            "distance should be preserved: before={d_before}, after={d_after}"
        );
    }

    #[test]
    fn test_translate_moves_anchor_neighborhood_near_origin() {
        // The actual point of recentering: points bunched near the boundary
        // (large r, where 1 - r² is tiny and precision degrades) should end
        // up close to the new origin after translation.
        let anchor = [0.9, 0.0, 0.0];
        let nearby = [0.91, 0.02, 0.01];

        let nearby_t = translate_to_origin(&anchor, &nearby);
        let norm_t: f64 = nearby_t.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Before: nearby has r ≈ 0.91, deep in the precision-loss zone.
        // After: should be much closer to r ≈ 0.
        assert!(norm_t < 0.2, "expected small radius after recentering, got {norm_t}");
    }

    #[test]
    fn test_distance_matches_mobius_formula() {
        // Independent cross-check, not just a self-consistency test: the
        // κ-stereographic model (Bloch/Skopek/Bachmann, ETH Zürich, building
        // on Ungar's gyrovector formalism) defines hyperbolic distance as
        //   d(x,y) = 2 · arctanh(‖(−x) ⊕ y‖)
        // which for our standard curvature-(-1) ball is exactly
        //   2 · atanh(‖translate_to_origin(x, y)‖)
        // Our poincare_distance uses the direct acosh closed form instead.
        // Same math, two independently-derived formulas — if they disagree,
        // one of our two functions has a real bug.
        let x = [0.3, 0.1, 0.2];
        let y = [0.5, 0.4, -0.1];

        let d_direct = poincare_distance(&x, &y);

        let translated = translate_to_origin(&x, &y);
        let norm: f64 = translated.iter().map(|v| v * v).sum::<f64>().sqrt();
        let d_mobius = 2.0 * norm.atanh();

        assert!(
            (d_direct - d_mobius).abs() < 1e-8,
            "distance formulas disagree: direct={d_direct}, mobius={d_mobius}"
        );
    }
}