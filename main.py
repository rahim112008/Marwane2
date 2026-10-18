"""
🐄 Bovine SNP Platform v4.0 (corrigée + GWAS + Convertisseur Axiom + Mapping + Outliers + cM)
Pipeline complet de bioinformatique pour puces SNP bovines.

Installation : pip install -r requirements.txt
Lancement   : streamlit run main.py
"""

import gzip
import io
import json
import os
import warnings
import zipfile
from collections import Counter
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🐄 Bovine SNP Platform v4.1",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_THRESHOLDS = {
    "geno": 0.05, "mind": 0.05, "maf": 0.05, "hwe": 1e-6, "het_sd": 3.0,
    "king_cutoff": 0.354, "ld_r2": 0.2,
}

HWE_EXACT_MAX_SNP = 20_000

ARS_UCD12_LENGTHS = {
    "1": 158_534_110, "2": 136_231_102, "3": 121_005_158, "4": 120_000_166,
    "5": 120_089_699, "6": 117_806_340, "7": 110_682_743, "8": 113_384_748,
    "9": 105_708_134, "10": 103_308_737, "11": 107_310_763, "12": 91_163_125,
    "13": 84_246_514, "14": 84_648_346, "15": 85_207_080, "16": 81_726_628,
    "17": 75_176_999, "18": 66_059_976, "19": 64_089_169, "20": 72_042_983,
    "21": 71_599_096, "22": 61_416_492, "23": 52_531_573, "24": 62_384_193,
    "25": 42_959_610, "26": 51_680_158, "27": 46_772_073, "28": 46_333_854,
    "29": 51_319_414, "X": 139_009_144, "Y": 50_927_933, "MT": 16_338,
}
BOVINE_AUTOSOMES = [str(i) for i in range(1, 30)]

# ============================================================
# CACHE STREAMLIT
# ============================================================

def _hash_ndarray(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return "empty"
    if x.dtype.kind in ("U", "S", "O"):
        preview = "|".join(map(str, x.ravel()[:2000]))
        return f"strarr|{x.shape}|{preview}"
    return (f"{x.shape}|{x.dtype}|"
            f"{float(np.nansum(x))}|"
            f"{float(np.nansum(np.abs(x)))}|"
            f"{float(np.nansum(x * x))}")


def _hash_any(x):
    try:
        arr = np.asarray(x)
        return _hash_ndarray(arr)
    except Exception:
        try:
            return f"{type(x).__name__}|{len(x)}"
        except Exception:
            return type(x).__name__


HASH_FUNCS = {
    np.ndarray: _hash_ndarray,
    pd.Series: _hash_any,
    pd.Index: _hash_any,
}

for _name in (
    "ArrowStringArray", "StringArray", "IntegerArray", "FloatingArray",
    "BooleanArray", "NumpyExtensionArray", "PandasArray",
    "DatetimeArray", "TimedeltaArray", "PeriodArray",
    "IntervalArray", "Categorical",
):
    _cls = getattr(pd.arrays, _name, None)
    if _cls is None:
        try:
            _cls = getattr(pd.core.arrays, _name, None)
        except Exception:
            _cls = None
    if _cls is not None:
        HASH_FUNCS[_cls] = _hash_any


def cache_data(func=None, **kw):
    def _decorate(f):
        return st.cache_data(show_spinner=False, hash_funcs=HASH_FUNCS, **kw)(f)
    if func is None:
        return _decorate
    return _decorate(func)


# ============================================================
# UTILITAIRES
# ============================================================

def impute_mean(gt):
    gt2 = gt.astype(np.float32, copy=True)
    col_mean = np.nanmean(gt2, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(gt2)
    if not nan_mask.any():
        return gt2
    gt2[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return gt2


def _autosome_weights():
    lens = np.array([ARS_UCD12_LENGTHS[c] for c in BOVINE_AUTOSOMES], dtype=float)
    return lens / lens.sum()


def _chr_sort_key(chrom):
    s = str(chrom).upper().replace("CHR", "").replace("CHROMOSOME", "")
    if s.isdigit():
        return (0, int(s), "")
    special = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    if s in special:
        return (1, special[s], "")
    return (2, 0, s)


def _detect_encoding(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        return "gzip"
    if len(raw) >= 2 and raw[:2] == b"\x6c\x1b":
        return "plink_binary"
    return "text"


def step_header(numero, titre, ce_qu_on_fait, pourquoi, attendu):
    with st.expander(f"📖 **Étape {numero} — {titre}**", expanded=False):
        st.markdown(f"**🎯 Ce qu'on fait :** {ce_qu_on_fait}")
        st.markdown(f"**❓ Pourquoi :** {pourquoi}")
        st.markdown(f"**✅ Ce qu'on attend :** {attendu}")


# ============================================================
# CONVERTISSEUR AXIOM → PED
# ============================================================

def sample_to_fid_iid(sample_name, fid_parts=2):
    name = str(sample_name).strip()
    for ext in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if name.endswith(ext):
            name = name[:-len(ext)]
    name = (name.replace("(", "_").replace(")", "")
                .replace("[", "_").replace("]", "")
                .replace(" ", "_").replace("/", "_"))
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")
    parts = name.split("_")
    fid = "_".join(parts[:fid_parts]) if len(parts) >= fid_parts else parts[0]
    return fid, name


def _clean_allele(a):
    if a is None:
        return "0"
    a = str(a).strip()
    if a in ("", ".", "-", "NA", "nan", "None", "0", "00"):
        return "0"
    return a


def parse_raw_axiom_file(raw_bytes, filename=""):
    if (filename.endswith(".gz")
            or (len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b")):
        try:
            raw = gzip.decompress(raw_bytes)
        except Exception as e:
            raise ValueError(f"Fichier .gz corrompu : {e}")
    else:
        raw = raw_bytes

    text = raw.decode("utf-8", errors="replace")
    lines = [l.rstrip("\r\n") for l in text.splitlines()]
    lines = [l for l in lines if l.strip()]
    if not lines:
        raise ValueError("Fichier vide.")

    first = lines[0]
    is_header = (first.startswith("#")
                 or first.startswith("Sample")
                 or "Sample Filename" in first[:50])
    start = 1 if is_header else 0

    samples, geno_rows = [], []
    for line in lines[start:]:
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
            if len(parts) < 2:
                continue
        sample_name = parts[0].strip()
        cells = [c.strip() for c in parts[1:]]
        alleles = []
        for cell in cells:
            toks = cell.split()
            if len(toks) == 2:
                alleles.append((_clean_allele(toks[0]),
                                _clean_allele(toks[1])))
            elif len(toks) == 1:
                s = toks[0]
                if len(s) == 2 and all(ch in "ACGT0" for ch in s):
                    alleles.append((s[0], s[1]))
                else:
                    alleles.append((_clean_allele(s), _clean_allele(s)))
            else:
                alleles.append(("0", "0"))
        samples.append(sample_name)
        geno_rows.append(alleles)

    if not samples:
        raise ValueError("Aucun échantillon trouvé dans le fichier.")

    counts = [len(r) for r in geno_rows]
    common = Counter(counts).most_common(1)[0][0]
    for i, r in enumerate(geno_rows):
        if len(r) > common:
            geno_rows[i] = r[:common]
        elif len(r) < common:
            geno_rows[i] = r + [("0", "0")] * (common - len(r))

    return samples, geno_rows, common


def build_ped_map_from_raw(samples, geno_rows, fid_parts=2):
    n_ind = len(samples)
    n_snp = len(geno_rows[0]) if n_ind else 0

    map_lines = [f"1\tSNP_{i+1:06d}\t0\t{i+1}" for i in range(n_snp)]
    map_text = "\n".join(map_lines) + "\n"

    ped_lines = []
    for i, sample in enumerate(samples):
        fid, iid = sample_to_fid_iid(sample, fid_parts=fid_parts)
        cols = [fid, iid, "0", "0", "0", "-9"]
        for a1, a2 in geno_rows[i]:
            cols.append(a1)
            cols.append(a2)
        ped_lines.append("\t".join(cols))
    ped_text = "\n".join(ped_lines) + "\n"

    return ped_text, map_text, n_snp


def preview_ped_dataframe(samples, geno_rows, max_rows=10, max_snp=20):
    rows = []
    n_snp = len(geno_rows[0]) if geno_rows else 0
    n_show = min(max_snp, n_snp)
    for sample, alleles in zip(samples[:max_rows], geno_rows[:max_rows]):
        fid, iid = sample_to_fid_iid(sample)
        row = {"FID": fid, "IID": iid, "PID": 0, "MID": 0,
               "Sex": 0, "Pheno": -9}
        for j in range(n_show):
            a1, a2 = alleles[j]
            row[f"SNP_{j+1:05d}_A1"] = a1
            row[f"SNP_{j+1:05d}_A2"] = a2
        rows.append(row)
    return pd.DataFrame(rows)


def render_axiom_converter_page():
    st.header("🔄 Convertisseur Axiom → PED")
    st.markdown("""
Transforme un fichier brut de génotypage Axiom/Thermo Fisher
(cellules `A1 A2` séparées par tabulation) en fichiers **PED** et **MAP**
au format PLINK.
    """)
    st.divider()

    raw_f = st.file_uploader(
        "Fichier brut Axiom (.ped, .txt, .tsv, .csv, .gz)",
        type=["ped", "txt", "tsv", "csv", "gz"],
        key="conv_raw_file")

    c1, c2 = st.columns(2)
    fid_parts = c1.number_input(
        "Nombre de parties du nom pour le FID",
        min_value=1, max_value=5, value=2, step=1, key="conv_fid_parts")
    sep_out = c2.selectbox(
        "Séparateur en sortie",
        ["Tabulation (PLINK standard)", "Espace"], key="conv_sep")

    if raw_f is not None:
        if st.button("🔄 Convertir", type="primary",
                     use_container_width=True, key="conv_btn_run"):
            try:
                with st.spinner("Parsing du fichier brut..."):
                    samples, geno_rows, n_snp = parse_raw_axiom_file(
                        raw_f.read(), raw_f.name)
                with st.spinner("Génération PED + MAP..."):
                    ped_text, map_text, _ = build_ped_map_from_raw(
                        samples, geno_rows, fid_parts=int(fid_parts))
                if sep_out.startswith("Espace"):
                    ped_text = ped_text.replace("\t", " ")
                    map_text = map_text.replace("\t", " ")
                st.session_state["_conv_samples"] = samples
                st.session_state["_conv_geno"] = geno_rows
                st.session_state["_conv_n_snp"] = n_snp
                st.session_state["_conv_ped"] = ped_text
                st.session_state["_conv_map"] = map_text
                st.success(f"✅ {len(samples)} individus × "
                           f"{n_snp} SNPs convertis.")
            except Exception as e:
                st.error(f"❌ {e}")

    if st.session_state.get("_conv_ped"):
        samples = st.session_state["_conv_samples"]
        geno_rows = st.session_state["_conv_geno"]
        n_snp = st.session_state["_conv_n_snp"]
        ped_text = st.session_state["_conv_ped"]
        map_text = st.session_state["_conv_map"]

        st.subheader("💾 Téléchargement")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇ Fichier .ped", data=ped_text.encode("utf-8"),
                file_name="converted.ped", mime="text/plain",
                use_container_width=True, key="conv_dl_ped")
        with c2:
            st.download_button(
                "⬇ Fichier .map", data=map_text.encode("utf-8"),
                file_name="converted.map", mime="text/plain",
                use_container_width=True, key="conv_dl_map")


# ============================================================
# PARSING MAP & PED
# ============================================================

def parse_map(map_bytes):
    enc = _detect_encoding(map_bytes)
    if enc == "gzip":
        text = gzip.decompress(map_bytes).decode("utf-8", errors="replace")
    else:
        text = map_bytes.decode("utf-8", errors="replace")

    rows, rejected = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            rejected += 1
            continue
        try:
            cm = float(parts[2]) if parts[2] not in ("0", ".", "NA") else 0.0
        except ValueError:
            cm = 0.0
        try:
            bp = int(float(parts[3]))
        except (ValueError, TypeError):
            rejected += 1
            continue
        rows.append({"CHR": str(parts[0]), "SNP": parts[1], "CM": cm, "BP": bp})

    if not rows:
        raise ValueError("Fichier .map vide ou invalide.")
    return pd.DataFrame(rows), rejected


def parse_ped(ped_bytes, n_snp_map):
    enc = _detect_encoding(ped_bytes)
    if enc == "gzip":
        text = gzip.decompress(ped_bytes).decode("utf-8", errors="replace")
    else:
        text = ped_bytes.decode("utf-8", errors="replace")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("Fichier .ped vide.")

    n_cols_first = len(lines[0].split())
    n_snp_ped = max(0, (n_cols_first - 6) // 2)
    n_snp_eff = min(n_snp_map, n_snp_ped)
    expected_cols_eff = 6 + 2 * n_snp_eff

    fids, iids, geno_rows = [], [], []
    for line in lines:
        parts = line.split()
        if len(parts) < expected_cols_eff:
            continue
        fids.append(parts[0])
        iids.append(parts[1])
        geno_rows.append(parts[6:6 + 2 * n_snp_eff])

    n_ind = len(iids)
    if n_ind == 0:
        raise ValueError("Aucun individu chargé.")

    allele_counts = [Counter() for _ in range(n_snp_eff)]
    for row in geno_rows:
        for j in range(n_snp_eff):
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 != "0" and a2 != "0":
                allele_counts[j][a1] += 1
                allele_counts[j][a2] += 1

    minor = [None] * n_snp_eff
    for j, c in enumerate(allele_counts):
        if len(c) >= 2:
            minor[j] = min(c, key=c.get)

    gt = np.full((n_ind, n_snp_eff), np.nan, dtype=np.float32)
    for i, row in enumerate(geno_rows):
        for j in range(n_snp_eff):
            m = minor[j]
            if m is None:
                continue
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0":
                continue
            gt[i, j] = (1 if a1 == m else 0) + (1 if a2 == m else 0)

    ind_df = pd.DataFrame({"FID": fids, "IID": iids})
    return gt, ind_df, 0


# ============================================================
# PARSING VCF & DEMO
# ============================================================

def parse_vcf(vcf_bytes):
    is_gz = len(vcf_bytes) >= 2 and vcf_bytes[:2] == b"\x1f\x8b"
    text = gzip.decompress(vcf_bytes).decode("utf-8", errors="replace") if is_gz else vcf_bytes.decode("utf-8", errors="replace")
    
    samples, rows = [], []
    for line in text.splitlines():
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
            continue
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 10: continue
        
        chrom, pos_s, vid, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
        if "," in alt: continue
        fmt = parts[8].split(":")
        if "GT" not in fmt: continue
        gt_idx = fmt.index("GT")

        gts = np.full(len(samples), np.nan, dtype=np.float32)
        for k, s in enumerate(parts[9:]):
            fields = s.split(":")
            if gt_idx < len(fields):
                gt_str = fields[gt_idx]
                sep = "|" if "|" in gt_str else "/"
                alleles = gt_str.split(sep)
                if not any(a in (".", "") for a in alleles):
                    gts[k] = sum(int(a) for a in alleles)

        rows.append({"CHR": str(chrom), "SNP": vid or f"{chrom}:{pos_s}",
                     "CM": 0.0, "BP": int(pos_s), "A1": ref, "A2": alt, "GT": gts})

    if not rows:
        raise ValueError("Aucun variant biallélique trouvé dans le VCF.")

    n_ind, n_snp = len(samples), len(rows)
    gt = np.zeros((n_ind, n_snp), dtype=np.float32)
    for j, r in enumerate(rows):
        gt[:, j] = r["GT"]
        valid = gt[:, j][~np.isnan(gt[:, j])]
        if len(valid) > 0 and valid.mean() / 2.0 > 0.5:
            gt[:, j] = np.where(np.isnan(gt[:, j]), np.nan, 2.0 - gt[:, j])

    snp_df = pd.DataFrame({
        "CHR": [r["CHR"] for r in rows], "SNP": [r["SNP"] for r in rows],
        "CM": 0.0, "BP": [r["BP"] for r in rows],
        "A1": [r["A1"] for r in rows], "A2": [r["A2"] for r in rows]})
    ind_df = pd.DataFrame({"FID": samples, "IID": samples})
    return gt, ind_df, snp_df, 0, 0


def generate_demo_data(n_ind=150, n_snp=800, n_pop=4, seed=42):
    rng = np.random.default_rng(seed)
    pool = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR", "NBD", "NKA", "PRS"]
    pop_names = pool[:min(n_pop, len(pool))]
    counts = [n_ind // n_pop + (1 if i < n_ind % n_pop else 0) for i in range(n_pop)]

    p_anc = rng.beta(0.6, 0.6, n_snp)
    gt = np.zeros((n_ind, n_snp), dtype=np.float32)
    ind_rows, k = [], 0
    for pop, n_i in zip(pop_names, counts):
        p_k = np.clip(p_anc + rng.normal(0, 0.15, n_snp), 0.02, 0.98)
        for i in range(n_i):
            gt[k] = (rng.random(n_snp) < p_k).astype(np.float32) + (rng.random(n_snp) < p_k).astype(np.float32)
            ind_rows.append({"FID": pop, "IID": f"{pop}_{i + 1:03d}"})
            k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan

    chr_names = rng.choice(BOVINE_AUTOSOMES, n_snp, p=_autosome_weights())
    bps = np.array([rng.integers(1, ARS_UCD12_LENGTHS[c]) for c in chr_names], dtype=np.int64)
    
    # Ajout de fausses distances génétiques (cM) pour la démo
    snp_df = pd.DataFrame({
        "CHR": chr_names, "SNP": [f"rs{i:07d}" for i in range(n_snp)],
        "CM": bps / 1_000_000, # Approximation 1Mb = 1cM
        "BP": bps})
    
    snp_df["_k"] = snp_df["CHR"].map(_chr_sort_key)
    snp_df = snp_df.sort_values(["_k", "BP"]).drop(columns="_k").reset_index(drop=True)
    return gt, pd.DataFrame(ind_rows), snp_df


# ============================================================
# QC & STATISTIQUES
# ============================================================

def missingness_per_ind(gt): return np.isnan(gt).mean(axis=1)
def missingness_per_snp(gt): return np.isnan(gt).mean(axis=0)
def allele_freq(gt): return np.nanmean(gt, axis=0) / 2.0
def maf(gt): return np.minimum(allele_freq(gt), 1.0 - allele_freq(gt))
def heterozygosity(gt): return np.nanmean(gt == 1, axis=1)

def hwe_exact_p(n_het, n_hom1, n_hom2):
    n = n_het + n_hom1 + n_hom2
    if n == 0 or n_het == 0 and (n_hom1 == 0 or n_hom2 == 0): return 1.0
    rare = 2 * min(n_hom1, n_hom2) + n_het
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2: mid += 1
    probs = np.zeros(rare + 1); probs[mid] = 1.0
    mysum, ch, chr_, chc = 1.0, mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        probs[ch + 2] = probs[ch] * 4 * chr_ * chc / ((ch + 2) * (ch + 1))
        mysum += probs[ch + 2]; ch += 2; chr_ -= 1; chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        probs[ch - 2] = probs[ch] * ch * (ch - 1) / (4 * (chr_ + 1) * (chc + 1))
        mysum += probs[ch - 2]; ch -= 2; chr_ += 1; chc += 1
    p_obs = probs[n_het] if n_het < len(probs) else 0.0
    return float(min(probs[probs <= p_obs + 1e-7].sum() / mysum, 1.0))

@cache_data
def hwe_pvalues(gt):
    _, n_snp = gt.shape
    pvals = np.full(n_snp, np.nan, dtype=np.float64)
    nan_mask = np.isnan(gt)
    n0 = ((gt == 0) & ~nan_mask).sum(axis=0)
    n1 = ((gt == 1) & ~nan_mask).sum(axis=0)
    n2 = ((gt == 2) & ~nan_mask).sum(axis=0)
    n_valid = n0 + n1 + n2

    if n_snp <= HWE_EXACT_MAX_SNP:
        for j in range(n_snp):
            if n_valid[j] >= 5: pvals[j] = hwe_exact_p(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(n_valid > 0, n_valid, np.nan))
        exp_het = 2.0 * p * (1.0 - p) * n_valid
        valid = (exp_het >= 5) & np.isfinite(exp_het)
        chi2 = (np.abs(n1 - exp_het) - 0.5) ** 2 / exp_het
        pvals[valid] = 1.0 - stats.chi2.cdf(chi2[valid], df=1)
    return pvals

def reconstruct_fid_from_iid(ind_df):
    df = ind_df.copy()
    if df["FID"].nunique() <= 1:
        df["FID"] = df["IID"].astype(str).str.split("_").str[0]
    return df

def filter_autosomes(gt, snp_df, keep_auto_only=True):
    chr_str = snp_df["CHR"].astype(str).str.upper().str.replace("CHR", "")
    keep = chr_str.isin([str(i) for i in range(1, 30)] + ([] if keep_auto_only else ["X", "Y", "MT"]))
    return gt[:, keep.values], snp_df[keep].reset_index(drop=True)

@cache_data
def ld_pruning(gt, window=50, step=5, r2_thr=0.2):
    n_snp = gt.shape[1]
    if n_snp < 3: return np.ones(n_snp, dtype=bool)
    keep = np.ones(n_snp, dtype=bool)
    X = impute_mean(gt); X = X - X.mean(axis=0)
    i = 0
    while i < n_snp:
        win_end = min(i + step, n_snp)
        while win_end < n_snp and (win_end - i) < window:
            j = win_end
            if keep[j] and keep[i]:
                a, b = X[:, i], X[:, j]
                if a.std() > 1e-8 and b.std() > 1e-8:
                    if np.corrcoef(a, b)[0, 1]**2 > r2_thr: keep[j] = False
            win_end += 1
        i += step
    return keep

def apply_qc_filters(gt, ind_df, snp_df, params, remove_outliers=True):
    n0, m0 = gt.shape
    trace = {}

    keep = missingness_per_snp(gt) <= params["geno"]
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["geno_exclus"] = int(m0 - gt.shape[1])

    keep = missingness_per_ind(gt) <= params["mind"]
    gt = gt[keep]; ind_df = ind_df[keep].reset_index(drop=True)
    trace["mind_exclus"] = int(n0 - gt.shape[0])

    m = maf(gt)
    keep = np.isfinite(m) & (m >= params["maf"])
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    
    pv = hwe_pvalues(gt)
    keep = np.isnan(pv) | (pv >= params["hwe"])
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)

    # --- SAUVEGARDE PRE-OUTLIERS ---
    gt_pre_outlier = gt.copy()
    ind_pre_outlier = ind_df.copy()
    snp_pre_outlier = snp_df.copy()

    if remove_outliers:
        het = heterozygosity(gt)
        keep = np.ones(len(het), dtype=bool)
        for fid in ind_df["FID"].unique():
            mask = (ind_df["FID"] == fid).values
            sub = het[mask]
            if len(sub) >= 5 and np.nanstd(sub) > 1e-9:
                z = (sub - np.nanmean(sub)) / np.nanstd(sub)
                idx = np.where(mask)[0]
                keep[idx[np.abs(z) > params["het_sd"]]] = False
        
        gt = gt[keep]; ind_df = ind_df[keep].reset_index(drop=True)
        trace["het_exclus"] = int(n0 - gt.shape[0])
    else:
        trace["het_exclus"] = 0

    return gt, ind_df, snp_df, gt_pre_outlier, ind_pre_outlier, snp_pre_outlier, {
        "n_ind_init": n0, "n_snp_init": m0,
        "n_ind_final": gt.shape[0], "n_snp_final": gt.shape[1],
        "excluded_ind": n0 - gt.shape[0], "excluded_snp": m0 - gt.shape[1],
        "trace": trace
    }

# ============================================================
# MODULES D'ANALYSE (FST, PCA, GWAS, etc.)
# ============================================================

@cache_data
def fst_per_snp(gt, pop_labels):
    pops = np.unique(np.asarray(pop_labels))
    if len(pops) < 2: return np.full(gt.shape[1], np.nan)
    fst = np.full(gt.shape[1], np.nan, dtype=np.float64)
    masks = {p: (np.asarray(pop_labels) == p) for p in pops}
    for j in range(gt.shape[1]):
        pl, nl = [], []
        for p in pops:
            vals = gt[masks[p], j]; vals = vals[~np.isnan(vals)]
            if len(vals) >= 3:
                pl.append(vals.mean() / 2.0); nl.append(len(vals))
        if len(pl) >= 2:
            p_bar = np.average(pl, weights=nl)
            h_s = np.average(2.0 * np.array(pl) * (1.0 - np.array(pl)), weights=nl)
            h_t = 2.0 * p_bar * (1.0 - p_bar)
            if h_t > 1e-9: fst[j] = (h_t - h_s) / h_t
    return fst

@cache_data
def pca_analysis(gt, n_components=10):
    X = impute_mean(gt); X = X - X.mean(axis=0)
    pca = PCA(n_components=min(n_components, X.shape[0] - 1, X.shape[1]))
    return pca.fit_transform(X), pca.explained_variance_ratio_ * 100.0

@cache_data
def selection_signatures(gt, snp_df, pop_labels):
    fst = fst_per_snp(gt, np.asarray(pop_labels))
    hom = np.nanmean((gt == 0) | (gt == 2), axis=0)
    df = snp_df.copy()
    df["FST"] = fst; df["HOM"] = hom
    hom_std = df["HOM"].std()
    df["SCORE"] = (df["FST"].fillna(0) * (df["HOM"] - df["HOM"].mean()) / (hom_std if hom_std > 1e-9 else 1.0))
    return df.sort_values("SCORE", ascending=False)

# ============================================================
# GWAS (LMM)
# ============================================================

def load_phenotypes_file(file_bytes, filename=""):
    if filename.endswith(".gz"): raw = gzip.decompress(file_bytes)
    else: raw = file_bytes
    text = raw.decode("utf-8", errors="replace")
    sep = "\t" if text.count("\t") > text.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def match_phenotypes(ind_df, pheno_df, iid_col=None, mapping_df=None):
    if mapping_df is not None:
        p = pheno_df.copy(); m = mapping_df.copy()
        if iid_col is None:
            for c in ["Animal", "IID", "id", "ID", "sample", "Sample"]:
                if c in p.columns: iid_col = c; break
        p[iid_col] = p[iid_col].astype(str)
        animal_col = [c for c in m.columns if c != "IID"][0]
        m[animal_col] = m[animal_col].astype(str); m["IID"] = m["IID"].astype(str)
        p_merged = p.merge(m, left_on=iid_col, right_on=animal_col, how="left")
        
        ind = ind_df.copy(); ind["_key"] = ind["IID"].astype(str)
        p_merged["_key"] = p_merged["IID"].astype(str)
        return ind[["_key", "FID", "IID"]].merge(p_merged.drop(columns=["IID"], errors="ignore"), on="_key", how="left").drop(columns="_key"), iid_col

    if iid_col is None:
        for c in ["IID", "id", "ID", "sample", "animal", "Animal"]:
            if c in pheno_df.columns: iid_col = c; break
    p = pheno_df.copy(); p[iid_col] = p[iid_col].astype(str)
    ind = ind_df.copy(); ind["_key"] = ind["IID"].astype(str); p["_key"] = p[iid_col]
    return ind[["_key", "FID", "IID"]].merge(p.drop(columns=[iid_col]), on="_key", how="left").drop(columns="_key"), iid_col

def _vc_estimate_delta(Y, X0, K):
    n = len(Y); d, U = np.linalg.eigh(K); d = np.maximum(d, 0)
    Yt = U.T @ Y; X0t = U.T @ X0
    def neg_loglik(delta):
        w = 1.0 / np.sqrt(delta * d + 1.0)
        beta, *_ = np.linalg.lstsq(X0t * w[:, None], Yt * w, rcond=None)
        s2 = np.sum((Yt * w - (X0t * w[:, None]) @ beta) ** 2) / n
        return -(-0.5 * (n * np.log(s2) + np.sum(np.log(delta * d + 1.0)))) if s2 > 0 else 1e18
    grid = np.logspace(-3, 3, 30)
    res = minimize_scalar(neg_loglik, bounds=(grid[np.argmin([neg_loglik(dl) for dl in grid])-1], grid[np.argmin([neg_loglik(dl) for dl in grid])+1]), method="bounded")
    return max(res.x, 1e-6)

@cache_data
def run_gwas_lmm(gt, pheno_vec, K, covariates, maf_min=0.05, missing_max=0.05, use_lmm=True):
    valid_ind = np.isfinite(pheno_vec)
    if valid_ind.sum() < 10: raise ValueError("Moins de 10 individus avec phénotype valide.")
    Y = pheno_vec[valid_ind]; gt_v = gt[valid_ind]; K_v = np.asarray(K)[np.ix_(valid_ind, valid_ind)]
    X0 = np.ones((len(Y), 1)) if covariates is None or covariates.size == 0 else np.column_stack([np.ones(len(Y)), np.asarray(covariates)[valid_ind]])
    X0 = X0[:, np.std(X0, axis=0) > 1e-10]
    n = len(Y)

    if use_lmm:
        delta = _vc_estimate_delta(Y, X0, K_v)
        d, U = np.linalg.eigh(K_v); d = np.maximum(d, 0)
        w = 1.0 / np.sqrt(delta * d + 1.0)
        Yt = (U.T @ Y) * w; X0t = (U.T @ X0) * w[:, None]
    else:
        Yt = Y; X0t = X0

    out = {"SNP_idx": [], "MAF": [], "BETA": [], "SE": [], "T": [], "P": []}
    for j in range(gt_v.shape[1]):
        g = gt_v[:, j].astype(float)
        if np.isnan(g).mean() > missing_max: continue
        g = np.where(np.isnan(g), np.nanmean(g), g)
        p = g.mean() / 2.0; maf_j = min(p, 1.0 - p)
        if maf_j < maf_min: continue

        X = np.column_stack([X0t, (U.T @ g) * w if use_lmm else g])
        beta, *_ = np.linalg.lstsq(X, Yt, rcond=None)
        resid = Yt - X @ beta
        s2 = float(np.sum(resid ** 2) / max(n - X.shape[1], 1))
        if s2 <= 0: continue
        
        se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1]))
        if se > 0:
            t_val = beta[-1] / se
            out["SNP_idx"].append(j); out["MAF"].append(maf_j); out["BETA"].append(beta[-1])
            out["SE"].append(se); out["T"].append(t_val); out["P"].append(float(2 * t_dist.sf(abs(t_val), df=max(n - X.shape[1], 1))))
    
    return pd.DataFrame(out)

def annotate_gwas_results(res_df, snp_df):
    if res_df.empty: return res_df
    idx = res_df["SNP_idx"].astype(int).values
    res_df = res_df.copy()
    res_df["SNP"] = snp_df["SNP"].astype(str).values[idx]
    res_df["CHR"] = snp_df["CHR"].astype(str).values[idx]
    res_df["BP"] = snp_df["BP"].astype(int).values[idx]
    res_df["CM"] = snp_df["CM"].astype(float).values[idx]
    return res_df[["SNP_idx", "SNP", "CHR", "BP", "CM", "MAF", "BETA", "SE", "T", "P"]]

def compute_lambda_gc(pvals):
    p = np.asarray(pvals); p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    return float(np.median(stats.chi2.isf(p, df=1)) / stats.chi2.ppf(0.5, df=1)) if len(p) >= 10 else np.nan

# ============================================================
# VISUALISATION
# ============================================================

def plot_hist(values, title, xlabel, color="#3498db"):
    fig = go.Figure(go.Histogram(x=values, nbinsx=80, marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Fréquence", height=380)
    return fig

def plot_pca(scores, var_pct, labels):
    df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "Population": labels})
    fig = px.scatter(df, x="PC1", y="PC2", color="Population", title=f"PCA — PC1 ({var_pct[0]:.1f}%) vs PC2 ({var_pct[1]:.1f}%)", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig

def plot_gwas_manhattan(res_df, bonferroni=None, fdr=None, title="GWAS Manhattan"):
    df = res_df[np.isfinite(res_df["P"])].copy()
    if df.empty: return None
    df["-log10P"] = -np.log10(np.clip(df["P"], 1e-300, 1))
    order = df["CHR"].map(_chr_sort_key)
    df = df.assign(_k0=order.map(lambda t: t[0]), _k1=order.map(lambda t: t[1]), _k2=order.map(lambda t: t[2])).sort_values(["_k0", "_k1", "BP"]).reset_index(drop=True)

    cum, ticks, labels = 0, [], []
    pos = np.zeros(len(df), dtype=int)
    for chrom in df["CHR"].unique():
        idx = np.where(df["CHR"] == chrom)[0]
        pos[idx] = np.arange(cum, cum + len(idx)); ticks.append((cum + cum + len(idx) - 1) / 2); labels.append(str(chrom)); cum += len(idx)
    df["x"] = pos

    fig = go.Figure()
    for i, chrom in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == chrom]
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["-log10P"], mode="markers", marker=dict(size=5, color="#2c3e50" if i % 2 == 0 else "#7f8c8d"),
            text=[f"{s}<br>CHR{c}:{b}<br>P={p:.2e}<br>β={bb:.3f}" for s, c, b, p, bb in zip(sub["SNP"], sub["CHR"], sub["BP"], sub["P"], sub["BETA"])], hoverinfo="text", showlegend=False))
    if bonferroni and np.isfinite(bonferroni): fig.add_hline(y=-np.log10(bonferroni), line_dash="dash", line_color="red", annotation_text=f"Bonferroni {bonferroni:.2e}")
    if fdr and np.isfinite(fdr): fig.add_hline(y=-np.log10(fdr), line_dash="dot", line_color="orange", annotation_text=f"FDR 5% {fdr:.2e}")
    fig.update_layout(title=title, xaxis_title="Chromosome", yaxis_title="-log10(P)", height=520, margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig

def plot_locuszoom(res_df, snp_df, chrom, center_bp, window_val, use_cm=False):
    df = res_df[res_df["CHR"].astype(str) == str(chrom)].copy()
    center_val = center_bp
    if use_cm:
        # Si on utilise les cM, on cherche le SNP le plus proche pour le centre
        sub = df.iloc[(df["CM"] - center_bp).abs().argsort()[:1]]
        center_val = sub["CM"].values[0] if not sub.empty else center_bp
        df = df[(df["CM"] >= center_val - window_val) & (df["CM"] <= center_val + window_val)]
        x_vals = df["CM"]
        x_label = "Position (cM)"
    else:
        window_bp = window_val * 1000
        df = df[(df["BP"] >= center_bp - window_bp) & (df["BP"] <= center_bp + window_bp)]
        x_vals = df["BP"] / 1e6
        x_label = "Position (Mb)"

    if df.empty: return None
    df["-log10P"] = -np.log10(np.clip(df["P"], 1e-300, 1))
    fig = go.Figure(go.Scatter(x=x_vals, y=df["-log10P"], mode="markers", marker=dict(size=8, color=df["-log10P"], colorscale="Viridis", showscale=True),
        text=[f"{s}<br>P={p:.2e}" for s, p in zip(df["SNP"], df["P"])], hoverinfo="text"))
    fig.update_layout(title=f"LocusZoom — CHR {chrom}", xaxis_title=x_label, yaxis_title="-log10(P)", height=500)
    return fig

# ============================================================
# INTERFACE STREAMLIT
# ============================================================

_STATE_KEYS = ["gt", "ind_df", "snp_df", "gt_filt", "ind_filt", "snp_filt", "gt_pre_outlier", "ind_pre_outlier", "snp_pre_outlier",
               "qc_stats", "gt_pruned", "snp_pruned", "pca_scores", "pca_var", "king_matrix", "fst", "admixture_Q", "admixture_K",
               "selection_df", "pheno_matched", "gwas_results", "gwas_pheno_name", "gwas_lambda"]

def init_state():
    for k in _STATE_KEYS:
        if k not in st.session_state: st.session_state[k] = None

def main():
    init_state()
    st.title("🐄 Bovine SNP Platform v4.1 (cM & Outliers)")
    
    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :", ["Demo", "Upload PED/MAP", "Upload VCF", "🔄 Convertir Axiom → PED"], key="sb_mode_source")

        if mode == "Demo":
            if st.button("🎲 Générer le jeu de démo", use_container_width=True):
                gt, ind_df, snp_df = generate_demo_data()
                st.session_state.gt = gt; st.session_state.ind_df = ind_df; st.session_state.snp_df = snp_df
                st.success("✅ Démo générée.")

        elif mode == "Upload PED/MAP":
            ped_f = st.file_uploader("Fichier .ped", type=["ped", "gz", "txt"])
            map_f = st.file_uploader("Fichier .map", type=["map", "gz", "txt"])
            if ped_f and map_f and st.button("📥 Charger PED + MAP"):
                map_df, _ = parse_map(map_f.read())
                gt, ind_df, _ = parse_ped(ped_f.read(), len(map_df))
                st.session_state.gt = gt; st.session_state.ind_df = ind_df; st.session_state.snp_df = map_df
                st.success("✅ Données chargées.")

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("Missingness SNP (--geno)", 0.0, 0.5, 0.05, 0.01)
        mind = st.slider("Missingness individu (--mind)", 0.0, 0.5, 0.05, 0.01)
        maf_thr = st.slider("MAF minimal (--maf)", 0.0, 0.5, 0.05, 0.01)
        hwe_thr = st.number_input("HWE p-value (--hwe)", value=1e-6, format="%.0e")
        het_sd = st.slider("Hétérozygotie ±σ intra-race", 1.0, 5.0, 3.0, 0.1)
        remove_outliers = st.checkbox("❌ Supprimer les outliers (Hétérozygotie)", value=True, help="Décochez pour garder tous les individus (ex: analyse de sélection).")
        ld_r2 = st.slider("LD Pruning r² seuil", 0.05, 0.5, 0.2, 0.05)

    if st.session_state.get("sb_mode_source") == "🔄 Convertir Axiom → PED":
        render_axiom_converter_page()
        return

    if st.session_state.gt is None:
        st.info("👉 Chargez des données ou générez un jeu de démo.")
        return

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "✂️ LD Pruning", "🧬 Structure", "🔍 Sélection", "🧪 GWAS"])

    # --- TAB QC ---
    with tabs[1]:
        if st.button("▶ Lancer le QC", type="primary"):
            params = {"geno": geno, "mind": mind, "maf": maf_thr, "hwe": hwe_thr, "het_sd": het_sd}
            gt_f, ind_f, snp_f, gt_pre, ind_pre, snp_pre, qc_stats = apply_qc_filters(
                st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, params, remove_outliers=remove_outliers)
            
            st.session_state.gt_filt = gt_f; st.session_state.ind_filt = ind_f; st.session_state.snp_filt = snp_f
            # SAUVEGARDE PRE-OUTLIERS
            st.session_state.gt_pre_outlier = gt_pre; st.session_state.ind_pre_outlier = ind_pre; st.session_state.snp_pre_outlier = snp_pre
            st.session_state.qc_stats = qc_stats
            st.success("✅ QC terminé.")

        if st.session_state.gt_filt is not None:
            s = st.session_state.qc_stats
            c1, c2, c3 = st.columns(3)
            c1.metric("Individus finaux", s["n_ind_final"], delta=-s["excluded_ind"])
            c2.metric("SNPs finaux", s["n_snp_final"], delta=-s["excluded_snp"])
            c3.metric("Outliers exclus", s["trace"].get("het_exclus", 0))
            
            st.info(f"💾 Jeu de données 'Pré-Outliers' sauvegardé : {st.session_state.gt_pre_outlier.shape[0]} individus, {st.session_state.gt_pre_outlier.shape[1]} SNPs.")

    # --- TAB LD PRUNING ---
    with tabs[2]:
        if st.session_state.gt_filt is not None:
            if st.button("▶ Lancer le LD Pruning"):
                keep = ld_pruning(st.session_state.gt_filt, window=50, step=5, r2_thr=ld_r2)
                st.session_state.gt_pruned = st.session_state.gt_filt[:, keep]
                st.session_state.snp_pruned = st.session_state.snp_filt[keep].reset_index(drop=True)
                st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs indépendants.")

    # --- TAB SÉLECTION (Avec option Outliers) ---
    with tabs[4]:
        st.subheader("🔍 Signatures de sélection (FST)")
        use_pre_outlier = st.checkbox("🧬 Utiliser le jeu de données AVANT suppression des outliers", value=True, key="sel_use_pre")
        
        if st.button("▶ Détecter les signatures"):
            if use_pre_outlier and st.session_state.gt_pre_outlier is not None:
                gt_sel = st.session_state.gt_pre_outlier
                snp_sel = st.session_state.snp_pre_outlier
                ind_sel = st.session_state.ind_pre_outlier
            else:
                gt_sel = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                snp_sel = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                ind_sel = st.session_state.ind_filt

            if gt_sel is not None:
                sel_df = selection_signatures(gt_sel, snp_sel, ind_sel["FID"].astype(str).values)
                st.session_state.selection_df = sel_df
                st.success(f"✅ Analyse réalisée sur {gt_sel.shape[0]} individus.")
                st.dataframe(sel_df.head(20))

    # --- TAB GWAS (Avec option cM) ---
    with tabs[5]:
        st.subheader("🧪 GWAS")
        
        # Mapping / Phenotypes upload code here (omitted for brevity, same as before)
        
        if st.session_state.get("pheno_matched") is not None:
            st.markdown("### Paramètres GWAS")
            c1, c2, c3 = st.columns(3)
            pheno_col = c1.selectbox("Phénotype", [c for c in st.session_state.pheno_matched.columns if c not in ("FID", "IID")])
            n_pcs = c2.slider("PCs comme covariables", 0, 10, 5)
            
            # OPTION 10 cM
            use_10cm = c3.checkbox("📏 Filtrer à 1 SNP / 10 cM", value=False, help="Nécessite une colonne 'CM' valide dans le fichier MAP.")
            
            if st.button("🚀 Lancer GWAS", type="primary"):
                gt_gwas = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                snp_gwas = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                
                if use_10cm:
                    if "CM" in snp_gwas.columns and snp_gwas["CM"].max() > 0:
                        # Garder 1 SNP par fenêtre de 10 cM
                        snp_gwas = snp_gwas.copy()
                        snp_gwas["_bin_cm"] = (snp_gwas["CM"] // 10).astype(int)
                        keep_idx = snp_gwas.groupby([ "CHR", "_bin_cm"]).head(1).index
                        gt_gwas = gt_gwas[:, keep_idx]
                        snp_gwas = snp_gwas.loc[keep_idx].reset_index(drop=True)
                        st.info(f"📏 Filtre 10 cM appliqué : {gt_gwas.shape[1]} SNPs conservés.")
                    else:
                        st.warning("⚠️ Impossible d'appliquer le filtre 10 cM : colonne 'CM' vide ou absente (toutes les valeurs sont à 0).")

                covs = st.session_state.pca_scores[:, :n_pcs] if n_pcs > 0 and st.session_state.pca_scores is not None else np.empty((gt_gwas.shape[0], 0))
                pheno_vec = st.session_state.pheno_matched[pheno_col].values
                K_use = np.eye(gt_gwas.shape[0]) # Remplacer par st.session_state.king_matrix si calculé
                
                res = run_gwas_lmm(gt_gwas, pheno_vec, K_use, covs)
                res = annotate_gwas_results(res, snp_gwas)
                st.session_state.gwas_results = res
                st.success(f"✅ {len(res)} SNPs testés.")

                fig = plot_gwas_manhattan(res)
                if fig: st.plotly_chart(fig, use_container_width=True)
                
                if not res.empty:
                    top_snp = res.sort_values("P").iloc[0]
                    st.markdown(f"### 🔍 LocusZoom (Lead SNP : {top_snp['SNP']})")
                    window_type = st.radio("Type de fenêtre", ["Physique (kb)", "Génétique (cM)"], horizontal=True)
                    win_val = st.slider("Taille de la fenêtre", 1, 50, 10) if window_type == "Génétique (cM)" else st.slider("Taille de la fenêtre (kb)", 50, 2000, 500)
                    
                    fig_lz = plot_locuszoom(res, snp_gwas, top_snp["CHR"], top_snp["CM"] if window_type == "Génétique (cM)" else top_snp["BP"], win_val, use_cm=(window_type == "Génétique (cM)"))
                    if fig_lz: st.plotly_chart(fig_lz, use_container_width=True)

if __name__ == "__main__":
    main()
