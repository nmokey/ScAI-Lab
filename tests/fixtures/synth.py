"""
Synthetic data factory.

The whole control-test strategy rests on being able to build embedding sets whose
signal content we *know exactly*. Everything here composes additive, mutually
orthogonal signal directions on top of isotropic Gaussian noise, so a fixture can
say "this dataset contains week and session structure but provably zero genotype
information" and have that be literally true rather than approximately true.

Design notes
------------
* Signal directions are drawn from a QR decomposition of a random matrix, so
  distinct factors (week / genotype / subject / session) are exactly orthogonal.
  A probe that recovers genotype from a `signal=("week",)` dataset is therefore
  recovering it from somewhere other than the embeddings.
* Subject IDs follow the real convention `{cohort}_{genotype}_{nn}`, with mouse
  numbering restarting per genotype, because five different modules parse that
  string and the fixtures must exercise the real format.
* Session structure mirrors the real acquisition: 1-4 mice are scanned in one
  volume and cropped apart afterwards, and per docs/DATA_MANIFEST.md the *same*
  group of same-genotype mice is scanned together at every week. So there are two
  distinct acquisition factors, both genotype-pure:
      "session" -- one (group, week) acquisition
      "group"   -- the cage/scan group, stable across all four weeks
  This is what makes the session-confound probe meaningful.
"""

from dataclasses import dataclass, field

import numpy as np

WEEK_ORDER = ["Week 12", "Week 15", "Week 18", "Week 20"]

# Named signal factors that can be planted in an embedding set.
FACTORS = ("week", "genotype", "subject", "session", "group")


@dataclass
class SynthEmbeddings:
    """A synthetic embedding set conforming to the standard .npz interface."""

    embeddings: np.ndarray  # (N, D) float32
    subject_ids: np.ndarray  # (N,) str
    weeks: np.ndarray  # (N,) str
    modalities: np.ndarray  # (N,) str
    paths: np.ndarray  # (N,) str
    session_ids: np.ndarray  # (N,) str -- NOT in the current .npz contract; see R5
    group_ids: np.ndarray  # (N,) str -- the cage/scan group, stable across weeks
    genotypes: np.ndarray  # (N,) str  "WT" | "KO"
    planted: tuple = field(default=())  # which factors actually carry signal

    def __len__(self):
        return len(self.embeddings)

    @property
    def genotype_binary(self):
        """1 = KO, 0 = WT — matching the convention used across the pipeline."""
        return (self.genotypes == "KO").astype(int)

    def subject_level(self, week="Week 12"):
        """One row per subject at a given week: (X, y_genotype, subject_ids)."""
        mask = self.weeks == week
        return (
            self.embeddings[mask],
            self.genotype_binary[mask],
            self.subject_ids[mask],
        )

    def save_npz(self, path):
        """Write in the exact format the encoder scripts emit and the evaluator reads."""
        np.savez(
            path,
            embeddings=self.embeddings,
            subject_ids=self.subject_ids,
            weeks=self.weeks,
            modalities=self.modalities,
            paths=self.paths,
        )
        return path


def _orthonormal_directions(n, dim, rng):
    """n exactly-orthonormal directions in R^dim (n <= dim)."""
    if n > dim:
        raise ValueError(f"cannot draw {n} orthogonal directions in {dim} dims")
    q, _ = np.linalg.qr(rng.standard_normal((dim, n)))
    return q.T  # (n, dim), rows orthonormal


def make_embeddings(
    n_subjects=32,
    cohort="NaF",
    weeks=WEEK_ORDER,
    dim=768,
    signal=(),
    strength=3.0,
    noise=1.0,
    session_size=4,
    missing=None,
    seed=0,
):
    """
    Build a synthetic embedding set with precisely controlled signal content.

    Parameters
    ----------
    n_subjects : int
        Split evenly between WT and KO.
    signal : str or sequence of str
        Which factors carry signal. Any of FACTORS. `()` (the default) gives pure
        noise — the negative-control dataset.
    strength : float
        Magnitude of each planted direction, relative to `noise` (per-component
        Gaussian sigma). The default of 3.0 against unit noise in 768 dims is a
        *weak* per-dimension effect but easily linearly separable, which is the
        regime the real embeddings sit in.
    session_size : int
        Mice per acquisition session. Sessions are genotype-pure and week-specific,
        matching the real study design.
    missing : dict or None
        Optional {subject_id: [weeks to drop]} to simulate incomplete attendance.
    """
    if isinstance(signal, str):
        signal = (signal,)
    unknown = set(signal) - set(FACTORS)
    if unknown:
        raise ValueError(f"unknown signal factor(s): {sorted(unknown)}")

    rng = np.random.default_rng(seed)

    # --- Build the subject roster: NaF_WT_01..., NaF_KO_01... ---
    n_wt = n_subjects // 2
    n_ko = n_subjects - n_wt
    subjects = [(f"{cohort}_WT_{i + 1:02d}", "WT") for i in range(n_wt)]
    subjects += [(f"{cohort}_KO_{i + 1:02d}", "KO") for i in range(n_ko)]

    # --- Assign sessions: a session scans one group of same-genotype mice at one week ---
    # This reproduces the real confound: genotype is constant within a session.
    group_of = {}
    for geno in ("WT", "KO"):
        same = [sid for sid, g in subjects if g == geno]
        for i, sid in enumerate(same):
            group_of[sid] = f"{cohort}_{geno}_g{i // session_size}"

    # --- Signal directions, exactly orthogonal across factors ---
    n_sessions = len({f"{g}@{w}" for g in group_of.values() for w in weeks})
    n_groups = len(set(group_of.values()))
    counts = {
        "week": len(weeks),
        "genotype": 2,
        "subject": len(subjects),
        "session": n_sessions,
        "group": n_groups,
    }
    total = sum(counts[f] for f in signal)
    dirs = _orthonormal_directions(total, dim, rng) if total else np.zeros((0, dim))

    offset, vecs = 0, {}
    for f in signal:
        vecs[f] = dirs[offset : offset + counts[f]]
        offset += counts[f]

    week_idx = {w: i for i, w in enumerate(weeks)}
    geno_idx = {"WT": 0, "KO": 1}
    subj_idx = {sid: i for i, (sid, _) in enumerate(subjects)}
    session_list = sorted({f"{g}@{w}" for g in group_of.values() for w in weeks})
    sess_idx = {s: i for i, s in enumerate(session_list)}
    group_list = sorted(set(group_of.values()))
    group_idx = {g: i for i, g in enumerate(group_list)}

    missing = missing or {}
    rows = []
    for sid, geno in subjects:
        for wk in weeks:
            if wk in missing.get(sid, []):
                continue
            session = f"{group_of[sid]}@{wk}"
            v = rng.standard_normal(dim) * noise
            if "week" in signal:
                v = v + strength * vecs["week"][week_idx[wk]]
            if "genotype" in signal:
                v = v + strength * vecs["genotype"][geno_idx[geno]]
            if "subject" in signal:
                v = v + strength * vecs["subject"][subj_idx[sid]]
            if "session" in signal:
                v = v + strength * vecs["session"][sess_idx[session]]
            if "group" in signal:
                v = v + strength * vecs["group"][group_idx[group_of[sid]]]
            rows.append((v, sid, wk, geno, session, group_of[sid]))

    return SynthEmbeddings(
        embeddings=np.stack([r[0] for r in rows]).astype(np.float32),
        subject_ids=np.array([r[1] for r in rows], dtype=str),
        weeks=np.array([r[2] for r in rows], dtype=str),
        modalities=np.array(["CT_HiRes"] * len(rows), dtype=str),
        paths=np.array([f"/synthetic/{r[1]}/{r[2].replace(' ', '_')}/ct_hi.nii.gz" for r in rows], dtype=str),
        session_ids=np.array([r[4] for r in rows], dtype=str),
        group_ids=np.array([r[5] for r in rows], dtype=str),
        genotypes=np.array([r[3] for r in rows], dtype=str),
        planted=tuple(signal),
    )


def make_tbr_table(emb, trend=0.0, ko_offset=0.0, base=20.0, noise=1.0, seed=0):
    """
    Synthetic TBR ground truth as a DataFrame-shaped list of dicts, matching the
    columns `create_mouse_traj_dataset.load_tbr` expects
    (subject_id, week, tbr2_p95_median).

    trend      : TBR increase per week-step (disease progression)
    ko_offset  : additive KO-vs-WT difference (0.0 = TBR carries no genotype signal)
    """
    rng = np.random.default_rng(seed)
    week_num = {"Week 12": 0, "Week 15": 1, "Week 18": 2, "Week 20": 3}
    rows = []
    for i in range(len(emb)):
        sid = str(emb.subject_ids[i])
        wk = str(emb.weeks[i])
        val = base + trend * week_num[wk] + rng.normal(0, noise)
        if emb.genotypes[i] == "KO":
            val += ko_offset
        rows.append({"subject_id": sid, "week": wk, "tbr2_p95_median": float(val)})
    return rows


def write_tbr_csv(rows, path):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "week", "tbr2_p95_median"])
        w.writeheader()
        w.writerows(rows)
    return path


def loso_auroc(X, y, groups, seed=42):
    """
    Leave-one-group-out logistic-regression AUROC.

    Deliberately a *separate, minimal* implementation from the pipeline's
    `_loso_logistic`: control tests must not inherit a bug from the code they are
    auditing. Kept simple on purpose — one classifier, no tuning.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut

    y = np.asarray(y)
    y_true, y_prob = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups=groups):
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X[tr], y[tr])
        y_prob.extend(clf.predict_proba(X[te])[:, 1])
        y_true.extend(y[te])
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def permutation_null(statistic, y, groups, n_perm=200, seed=0):
    """
    Null distribution of `statistic(y_shuffled)` under label permutation.

    Labels are shuffled *at the group level* — a subject keeps one genotype across
    all its scans — because shuffling per-row would break the very dependence
    structure the null needs to preserve.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    y = np.asarray(y)

    uniq = np.unique(groups)
    per_group = np.array([y[groups == g][0] for g in uniq])

    null = []
    for _ in range(n_perm):
        shuffled = rng.permutation(per_group)
        mapping = dict(zip(uniq, shuffled))
        y_perm = np.array([mapping[g] for g in groups])
        val = statistic(y_perm)
        if val == val:  # drop NaN folds
            null.append(val)
    return np.array(null)
