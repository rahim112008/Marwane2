"""
🐄 Bovine SNP Platform v4.0 (corrigée + GWAS + Convertisseur Axiom)
Pipeline complet de bioinformatique pour puces SNP bovines.
Conforme au MANUSCRIT DE FORMATION.

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
    page_title="🐄 Bovine SNP Platform v4.0",
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
    """Hash d'un ndarray → STRING (évite la récursion infinie)."""
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

**Format d'entrée :**
#Sample Filename Genotypes
BEN_EBG_..._A01.CEL C C G G A G C C ...

text
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
                import traceback
                st.code(traceback.format_exc())

    if st.session_state.get("_conv_ped"):
        samples = st.session_state["_conv_samples"]
        geno_rows = st.session_state["_conv_geno"]
        n_snp = st.session_state["_conv_n_snp"]
        ped_text = st.session_state["_conv_ped"]
        map_text = st.session_state["_conv_map"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", len(samples))
        c2.metric("SNPs", n_snp)
        c3.metric("Colonnes PED", 6 + 2 * n_snp)

        st.subheader("👁️ Aperçu (10 lignes × 20 SNPs)")
        df_prev = preview_ped_dataframe(samples, geno_rows)
        st.dataframe(df_prev, use_container_width=True,
                     key="conv_df_preview")

        with st.expander("📄 Aperçu brut PED (5 lignes)"):
            st.code("\n".join(ped_text.splitlines()[:5]), language="text")
        with st.expander("📄 Aperçu MAP (10 premières lignes)"):
            st.code("\n".join(map_text.splitlines()[:10]), language="text")

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
        with c3:
            bio = BytesIO()
            with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("converted.ped", ped_text)
                zf.writestr("converted.map", map_text)
            st.download_button(
                "⬇ .zip (PED + MAP)", data=bio.getvalue(),
                file_name="converted_ped_map.zip", mime="application/zip",
                use_container_width=True, key="conv_dl_zip")

        st.info("💡 Ensuite, choisis la source **« Upload PED/MAP »** dans la "
                "sidebar pour charger ces fichiers et lancer le pipeline "
                "complet (QC → LD Pruning → PCA → GWAS...).")


# ============================================================
# PARSING MAP
# ============================================================

def parse_map(map_bytes):
    enc = _detect_encoding(map_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(map_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .map.gz corrompu : {e}")
    elif enc == "plink_binary":
        raise ValueError("❌ Le fichier .map est en binaire PLINK.")
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


# ============================================================
# PARSING PED
# ============================================================

def _diagnose_ped_first_line(line, n_snp_map):
    parts = line.split()
    n_cols = len(parts)
    return {
        "n_cols": n_cols,
        "expected_cols": 6 + 2 * n_snp_map,
        "n_snp_inferred": max(0, (n_cols - 6) // 2),
    }


def parse_ped(ped_bytes, n_snp_map):
    enc = _detect_encoding(ped_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(ped_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .ped.gz corrompu : {e}")
    elif enc == "plink_binary":
        raise ValueError("❌ Ce fichier est un PLINK binaire (.bed) renommé.")
    else:
        text = ped_bytes.decode("utf-8", errors="replace")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("Fichier .ped vide.")

    diag = _diagnose_ped_first_line(lines[0], n_snp_map)
    n_cols_first = diag["n_cols"]
    n_snp_ped = diag["n_snp_inferred"]

    if n_cols_first < 7:
        raise ValueError(f"❌ Format PED invalide : {n_cols_first} colonnes.")

    if 0 < n_snp_ped < 10 and len(lines) > 100:
        raise ValueError("⚠️ Fichier probablement transposé.")

    if n_snp_ped != n_snp_map and n_snp_ped > 0:
        st.warning(f"⚠️ Désalignement PED/MAP : {n_snp_ped} vs {n_snp_map} SNPs.")
        n_snp_eff = min(n_snp_map, n_snp_ped)
    else:
        n_snp_eff = n_snp_map

    expected_cols_eff = 6 + 2 * n_snp_eff
    fids, iids, geno_rows = [], [], []
    rejected_short, rejected_empty = 0, 0

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            rejected_empty += 1
            continue
        if len(parts) < expected_cols_eff:
            rejected_short += 1
            continue
        fids.append(parts[0])
        iids.append(parts[1])
        geno_rows.append(parts[6:6 + 2 * n_snp_eff])

    n_ind = len(iids)
    if n_ind == 0:
        raise ValueError(
            f"❌ Aucun individu chargé. Colonnes attendues : {expected_cols_eff}, "
            f"1ère ligne : {n_cols_first}")

    if rejected_short > 0:
        st.warning(f"⚠️ {rejected_short} ligne(s) ignorée(s).")

    allele_counts = [Counter() for _ in range(n_snp_eff)]
    for row in geno_rows:
        for j in range(n_snp_eff):
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0":
                continue
            c = allele_counts[j]
            c[a1] += 1
            c[a2] += 1

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
    return gt, ind_df, rejected_short + rejected_empty


# ============================================================
# PARSING VCF
# ============================================================

def parse_vcf(vcf_bytes):
    is_gz = len(vcf_bytes) >= 2 and vcf_bytes[:2] == b"\x1f\x8b"
    if is_gz:
        try:
            text = gzip.decompress(vcf_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .vcf.gz corrompu : {e}")
    else:
        text = vcf_bytes.decode("utf-8", errors="replace")

    samples, rows = [], []
    n_skipped, n_multi = 0, 0

    for line in text.splitlines():
        line = line.rstrip("\n\r")
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
            continue
        if line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 10:
            n_skipped += 1
            continue

        chrom, pos_s, vid, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
        if "," in alt:
            n_multi += 1
            continue
        try:
            pos = int(pos_s)
        except ValueError:
            n_skipped += 1
            continue

        fmt = parts[8].split(":")
        if "GT" not in fmt:
            n_skipped += 1
            continue
        gt_idx = fmt.index("GT")

        gts = np.full(len(samples), np.nan, dtype=np.float32)
        for k, s in enumerate(parts[9:]):
            fields = s.split(":")
            if gt_idx >= len(fields):
                continue
            gt_str = fields[gt_idx]
            sep = "|" if "|" in gt_str else "/"
            alleles = gt_str.split(sep)
            if any(a in (".", "") for a in alleles):
                continue
            try:
                gts[k] = sum(int(a) for a in alleles)
            except ValueError:
                continue

        rows.append({"CHR": str(chrom), "SNP": vid or f"{chrom}:{pos}",
                     "CM": 0.0, "BP": pos, "A1": ref, "A2": alt, "GT": gts})

    if not rows:
        raise ValueError("Aucun variant biallélique trouvé dans le VCF.")

    n_ind, n_snp = len(samples), len(rows)
    gt = np.zeros((n_ind, n_snp), dtype=np.float32)
    for j, r in enumerate(rows):
        gt[:, j] = r["GT"]

    for j in range(n_snp):
        col = gt[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        if valid.mean() / 2.0 > 0.5:
            gt[:, j] = np.where(np.isnan(col), np.nan, 2.0 - col)

    snp_df = pd.DataFrame({
        "CHR": [r["CHR"] for r in rows], "SNP": [r["SNP"] for r in rows],
        "CM": 0.0, "BP": [r["BP"] for r in rows],
        "A1": [r["A1"] for r in rows], "A2": [r["A2"] for r in rows]})
    ind_df = pd.DataFrame({"FID": samples, "IID": samples})
    return gt, ind_df, snp_df, n_skipped, n_multi


# ============================================================
# DONNÉES DE DÉMONSTRATION
# ============================================================

def generate_demo_data(n_ind=150, n_snp=800, n_pop=4, seed=42):
    rng = np.random.default_rng(seed)
    pool = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR",
            "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"]
    n_pop = max(1, min(n_pop, len(pool)))
    pop_names = pool[:n_pop]

    per_pop = n_ind // n_pop
    remainder = n_ind - per_pop * n_pop
    counts = [per_pop + (1 if i < remainder else 0) for i in range(n_pop)]
    n_ind_eff = sum(counts)

    p_anc = rng.beta(0.6, 0.6, n_snp)
    drift = 0.15
    gt = np.zeros((n_ind_eff, n_snp), dtype=np.float32)
    ind_rows = []
    k = 0
    for pop, n_i in zip(pop_names, counts):
        p_k = np.clip(p_anc + rng.normal(0, drift, n_snp), 0.02, 0.98)
        for i in range(n_i):
            a1 = (rng.random(n_snp) < p_k).astype(np.float32)
            a2 = (rng.random(n_snp) < p_k).astype(np.float32)
            gt[k] = a1 + a2
            ind_rows.append({"FID": pop, "IID": f"{pop}_{i + 1:03d}"})
            k += 1

    gt[rng.random(gt.shape) < 0.02] = np.nan

    ind_df = pd.DataFrame(ind_rows)
    chr_names = rng.choice(BOVINE_AUTOSOMES, n_snp, p=_autosome_weights())
    bps = np.array([rng.integers(1, ARS_UCD12_LENGTHS[c]) for c in chr_names],
                   dtype=np.int64)
    snp_df = pd.DataFrame({
        "CHR": chr_names, "SNP": [f"rs{i:07d}" for i in range(n_snp)],
        "CM": 0.0, "BP": bps})
    snp_df["_k"] = snp_df["CHR"].map(_chr_sort_key)
    snp_df = (snp_df.sort_values(["_k", "BP"]).drop(columns="_k")
              .reset_index(drop=True))
    return gt, ind_df, snp_df


# ============================================================
# MÉTRIQUES QC
# ============================================================

def missingness_per_ind(gt):
    return np.isnan(gt).mean(axis=1)


def missingness_per_snp(gt):
    return np.isnan(gt).mean(axis=0)


def allele_freq(gt):
    return np.nanmean(gt, axis=0) / 2.0


def maf(gt):
    p = allele_freq(gt)
    return np.minimum(p, 1.0 - p)


def heterozygosity(gt):
    with np.errstate(invalid="ignore"):
        return np.nanmean(gt == 1, axis=1)


def hwe_exact_p(n_het, n_hom1, n_hom2):
    n = n_het + n_hom1 + n_hom2
    if n == 0:
        return np.nan
    if n_het == 0 and (n_hom1 == 0 or n_hom2 == 0):
        return 1.0
    rare = 2 * min(n_hom1, n_hom2) + n_het
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2:
        mid += 1
    probs = np.zeros(rare + 1)
    probs[mid] = 1.0
    mysum = 1.0
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        probs[ch + 2] = probs[ch] * 4 * chr_ * chc / ((ch + 2) * (ch + 1))
        mysum += probs[ch + 2]
        ch += 2; chr_ -= 1; chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        probs[ch - 2] = probs[ch] * ch * (ch - 1) / (4 * (chr_ + 1) * (chc + 1))
        mysum += probs[ch - 2]
        ch -= 2; chr_ += 1; chc += 1
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
            if n_valid[j] < 5:
                continue
            pvals[j] = hwe_exact_p(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(n_valid > 0, n_valid, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_het = 2.0 * p * (1.0 - p) * n_valid
            valid = (exp_het >= 5) & np.isfinite(exp_het)
            chi2 = (np.abs(n1 - exp_het) - 0.5) ** 2 / exp_het
            pvals[valid] = 1.0 - stats.chi2.cdf(chi2[valid], df=1)
    return pvals


# ============================================================
# MODULE 2A — NETTOYAGE PED
# ============================================================

def extract_fid_from_iid(iid):
    for sep in ("_", "-", "."):
        if sep in iid:
            return iid.split(sep)[0]
    return iid


def reconstruct_fid_from_iid(ind_df):
    df = ind_df.copy()
    unique_fids = df["FID"].nunique() if "FID" in df.columns else 0
    if unique_fids <= 1:
        df["FID"] = df["IID"].astype(str).apply(extract_fid_from_iid)
        st.info(f"🔧 FID reconstruit depuis IID → **{df['FID'].nunique()}** races.")
    return df


def filter_autosomes(gt, snp_df, keep_auto_only=True):
    chr_str = snp_df["CHR"].astype(str).str.upper().str.replace("CHR", "")
    if keep_auto_only:
        keep = chr_str.isin([str(i) for i in range(1, 30)])
    else:
        keep = chr_str.isin([str(i) for i in range(1, 30)] + ["X", "Y", "MT"])
    n_before = len(snp_df)
    gt2 = gt[:, keep.values]
    snp2 = snp_df[keep].reset_index(drop=True)
    if keep_auto_only:
        st.info(f"🧬 Autosomes 1–29 : {n_before - len(snp2)} SNPs exclus → "
                f"**{len(snp2)}** conservés.")
    return gt2, snp2


# ============================================================
# MODULE 4.5 — LD PRUNING
# ============================================================

@cache_data
def ld_pruning(gt, window=50, step=5, r2_thr=0.2):
    n_snp = gt.shape[1]
    if n_snp < 3:
        return np.ones(n_snp, dtype=bool)

    keep = np.ones(n_snp, dtype=bool)
    X = impute_mean(gt)
    X = X - X.mean(axis=0)

    i = 0
    while i < n_snp:
        win_end = min(i + step, n_snp)
        while win_end < n_snp and (win_end - i) < window:
            j = win_end
            if keep[j] and keep[i]:
                a, b = X[:, i], X[:, j]
                if a.std() > 1e-8 and b.std() > 1e-8:
                    r = np.corrcoef(a, b)[0, 1]
                    if r * r > r2_thr:
                        keep[j] = False
            win_end += 1
        i += step
    return keep


# ============================================================
# QC — FILTRAGE PRINCIPAL
# ============================================================

def _align_shapes(gt, ind_df, snp_df):
    n_gt, m_gt = gt.shape
    n_ind, n_snp = min(n_gt, len(ind_df)), min(m_gt, len(snp_df))
    if (n_gt, m_gt) != (len(ind_df), len(snp_df)):
        st.warning(f"⚠️ Alignement : gt={gt.shape} → ({n_ind},{n_snp})")
    return (gt[:n_ind, :n_snp],
            ind_df.iloc[:n_ind].reset_index(drop=True),
            snp_df.iloc[:n_snp].reset_index(drop=True))


def apply_qc_filters(gt, ind_df, snp_df, params):
    gt, ind_df, snp_df = _align_shapes(gt, ind_df, snp_df)
    n0, m0 = gt.shape
    trace = {}

    keep = missingness_per_snp(gt) <= params["geno"]
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["geno_exclus"] = int(m0 - gt.shape[1])

    keep = missingness_per_ind(gt) <= params["mind"]
    gt = gt[keep]; ind_df = ind_df[keep].reset_index(drop=True)
    trace["mind_exclus"] = int(n0 - gt.shape[0])

    if gt.shape[1] == 0 or gt.shape[0] == 0:
        raise ValueError("Tous les SNPs ou individus exclus (missingness).")

    m_before = gt.shape[1]
    m = maf(gt)
    keep = np.isfinite(m) & (m >= params["maf"])
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["maf_exclus"] = int(m_before - gt.shape[1])

    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par MAF.")

    m_before = gt.shape[1]
    pv = hwe_pvalues(gt)
    keep = np.isnan(pv) | (pv >= 1e-6)
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    if gt.shape[1] > 0:
        pv2 = hwe_pvalues(gt)
        keep = np.isnan(pv2) | (pv2 >= params["hwe"])
        gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["hwe_exclus"] = int(m_before - gt.shape[1])

    if gt.shape[1] == 0:
        raise ValueError("Tous les SNPs exclus par HWE.")

    het = heterozygosity(gt)
    keep = np.ones(len(het), dtype=bool)
    for fid in ind_df["FID"].unique():
        mask = (ind_df["FID"] == fid).values
        sub = het[mask]
        if len(sub) < 5 or np.nanstd(sub) < 1e-9:
            continue
        z = (sub - np.nanmean(sub)) / np.nanstd(sub)
        sub_keep = np.abs(z) <= params["het_sd"]
        idx = np.where(mask)[0]
        keep[idx[~sub_keep]] = False
    n_before = gt.shape[0]
    gt = gt[keep]; ind_df = ind_df[keep].reset_index(drop=True)
    trace["het_exclus"] = int(n_before - gt.shape[0])

    if gt.shape[0] == 0:
        raise ValueError("Tous les individus exclus (hétérozygotie).")

    return gt, ind_df, snp_df, {
        "n_ind_init": int(n0), "n_snp_init": int(m0),
        "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]),
        "excluded_ind": int(n0 - gt.shape[0]),
        "excluded_snp": int(m0 - gt.shape[1]),
        "trace": trace,
    }


# ============================================================
# POPULATION GENETICS
# ============================================================

@cache_data
def fst_per_snp(gt, pop_labels):
    pops = np.unique(np.asarray(pop_labels))
    if len(pops) < 2:
        return np.full(gt.shape[1], np.nan)
    fst = np.full(gt.shape[1], np.nan, dtype=np.float64)
    masks = {p: (np.asarray(pop_labels) == p) for p in pops}
    for j in range(gt.shape[1]):
        pl, nl = [], []
        for p in pops:
            vals = gt[masks[p], j]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 3:
                continue
            pl.append(vals.mean() / 2.0); nl.append(len(vals))
        if len(pl) < 2:
            continue
        p_arr, n_arr = np.asarray(pl), np.asarray(nl)
        p_bar = np.average(p_arr, weights=n_arr)
        h_s = np.average(2.0 * p_arr * (1.0 - p_arr), weights=n_arr)
        h_t = 2.0 * p_bar * (1.0 - p_bar)
        if h_t > 1e-9:
            fst[j] = (h_t - h_s) / h_t
    return fst


@cache_data
def fst_pairwise(gt, pop_labels):
    labels = np.asarray(pop_labels)
    pops = sorted(np.unique(labels))
    K = len(pops)
    matrix = np.full((K, K), np.nan); np.fill_diagonal(matrix, 0.0)
    masks = {p: (labels == p) for p in pops}
    for i in range(K):
        for j in range(i + 1, K):
            g1 = gt[masks[pops[i]]]; g2 = gt[masks[pops[j]]]
            p1 = np.nanmean(g1, 0) / 2.0; p2 = np.nanmean(g2, 0) / 2.0
            n1 = (~np.isnan(g1)).sum(0); n2 = (~np.isnan(g2)).sum(0)
            denom = np.where(n1 + n2 > 0, n1 + n2, np.nan)
            p_bar = (p1 * n1 + p2 * n2) / denom
            hs = (2 * p1 * (1 - p1) * n1 + 2 * p2 * (1 - p2) * n2) / denom
            ht = 2 * p_bar * (1 - p_bar)
            with np.errstate(divide="ignore", invalid="ignore"):
                fst_j = (ht - hs) / ht
            matrix[i, j] = matrix[j, i] = float(np.nanmean(fst_j))
    return matrix, pops


@cache_data
def pca_analysis(gt, n_components=10):
    X = impute_mean(gt); X = X - X.mean(axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(X), pca.explained_variance_ratio_ * 100.0


@cache_data
def mds_analysis(gt, n_components=5):
    X = impute_mean(gt); n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    return SklearnMDS(n_components=n_components, dissimilarity="precomputed",
                      random_state=42, n_init=1, max_iter=300,
                      normalized_stress=False).fit_transform(D)


@cache_data
def ld_decay(gt, snp_bp, max_kb=1000, max_snp=1500, seed=42):
    n_snp = gt.shape[1]
    if n_snp < 2:
        return pd.DataFrame(columns=["dist_kb", "r2"])
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n_snp, max_snp, replace=False))
           if n_snp > max_snp else np.arange(n_snp))
    gt_sub = impute_mean(gt[:, idx]); bp_sub = np.asarray(snp_bp)[idx].astype(np.float64)
    X = gt_sub - gt_sub.mean(0)
    std = gt_sub.std(0); std[std < 1e-8] = np.nan
    X = X / std
    C = (X.T @ X) / X.shape[0]; R2 = C ** 2
    iu, ju = np.triu_indices(len(idx), k=1)
    dist_kb = (bp_sub[ju] - bp_sub[iu]) / 1000.0
    mask = (dist_kb > 0) & (dist_kb <= max_kb)
    return pd.DataFrame({"dist_kb": dist_kb[mask], "r2": R2[iu[mask], ju[mask]]})


@cache_data
def kinship_matrix(gt):
    X = impute_mean(gt)
    p = np.clip(X.mean(0) / 2.0, 1e-3, 1 - 1e-3)
    Z = X - 2 * p
    Zn = Z / np.sqrt(2 * p * (1 - p))
    return (Zn @ Zn.T) / X.shape[1]


@cache_data
def king_kinship(gt):
    n_ind = gt.shape[0]
    het = np.nansum(gt == 1, axis=1).astype(np.float64)
    K = np.zeros((n_ind, n_ind))
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            both_het = np.nansum((gt[i] == 1) & (gt[j] == 1))
            denom = het[i] + het[j]
            phi = (denom - 2 * both_het) / denom if denom > 0 else 0.0
            K[i, j] = K[j, i] = phi
    np.fill_diagonal(K, 0.5)
    return K


def apply_king_cutoff(K, ind_df, cutoff=0.354):
    n = K.shape[0]
    to_remove = np.zeros(n, dtype=bool)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if K[i, j] > cutoff and not to_remove[i] and not to_remove[j]]
    for i, j in pairs:
        to_remove[j] = True
    return ~to_remove, int(to_remove.sum())


@cache_data
def admixture_nmf(gt, K=3, seed=42, max_iter=500):
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    model = NMF(n_components=K, init="nndsvda",
                random_state=seed, max_iter=max_iter)
    W = model.fit_transform(X)
    s = W.sum(1, keepdims=True); s[s == 0] = 1.0
    return W / s, model.components_


@cache_data
def admix_cv_error(gt, K_range=(2, 6), n_reps=3, holdout=0.05, seed=42):
    rng = np.random.default_rng(seed)
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    n, m = X.shape
    mask = rng.random((n, m)) < holdout
    X_train = X.copy(); X_train[mask] = 0.0
    results = {}
    for K in range(K_range[0], K_range[1]):
        errs = []
        for rep in range(n_reps):
            model = NMF(n_components=K, init="nndsvda",
                        random_state=seed + rep, max_iter=300)
            W = model.fit_transform(X_train); H = model.components_
            err = np.mean((X[mask] - (W @ H)[mask]) ** 2)
            errs.append(err)
        results[K] = {"mean": float(np.mean(errs)),
                      "std": float(np.std(errs))}
    return results


@cache_data
def estimate_ne_historical(gt, snp_df, max_snp=2000):
    bp = snp_df["BP"].astype(np.float64).values
    chr_arr = snp_df["CHR"].astype(str).values
    n_snp = gt.shape[1]
    if n_snp > max_snp:
        idx = np.linspace(0, n_snp - 1, max_snp).astype(int)
        gt = gt[:, idx]; bp = bp[idx]; chr_arr = chr_arr[idx]

    X = impute_mean(gt)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    r2_list, dist_list = [], []
    n = X.shape[1]
    for i in range(0, n - 1):
        for j in range(i + 1, min(i + 50, n)):
            if chr_arr[i] != chr_arr[j]:
                continue
            d = abs(bp[j] - bp[i]) / 1e6
            if d < 0.001 or d > 50:
                continue
            r = np.corrcoef(X[:, i], X[:, j])[0, 1]
            r2_list.append(r * r); dist_list.append(d)

    if not dist_list:
        return pd.DataFrame({"GenAgo": [], "Ne": []})

    r2 = np.array(r2_list); dist = np.array(dist_list)
    bins = np.logspace(np.log10(dist.min() + 1e-6),
                       np.log10(dist.max()), 20)
    bc = (bins[:-1] + bins[1:]) / 2
    r2b = np.array([
        r2[(dist >= bins[k]) & (dist < bins[k + 1])].mean()
        if ((dist >= bins[k]) & (dist < bins[k + 1])).sum() > 0 else np.nan
        for k in range(len(bins) - 1)])
    m = np.isfinite(r2b)
    if m.sum() < 3:
        return pd.DataFrame({"GenAgo": [], "Ne": []})

    c_morgans = bc[m] * 0.01
    r2v = r2b[m]
    ne = np.where(r2v > 1e-4, 1.0 / (4 * c_morgans * r2v), np.nan)
    gen_ago = bc[m] * 100
    df = pd.DataFrame({"GenAgo": gen_ago, "Ne": ne}).dropna()
    df = df[df["Ne"] > 0].sort_values("GenAgo")
    if len(df) > 3:
        df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df


@cache_data
def reynolds_distance(gt, pop_labels):
    labels = np.asarray(pop_labels)
    pops = sorted(np.unique(labels)); K = len(pops)
    p_hat = {}
    for p in pops:
        p_hat[p] = np.nanmean(gt[labels == p], 0) / 2.0
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            pi, pj = p_hat[pops[i]], p_hat[pops[j]]
            num = np.nansum((pi - pj) ** 2)
            denom = np.nansum(pi * (1 - pj) + pj * (1 - pi))
            if denom > 0:
                D[i, j] = D[j, i] = num / denom
    return D, pops


def nj_tree_newick(D, labels):
    try:
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform
        Dm = squareform(D, checks=False)
        Z = linkage(Dm, method="average")
        tree = to_tree(Z)

        def rec(node):
            if node.is_leaf():
                return f"{labels[node.id]}:{node.dist:.4f}"
            return f"({rec(node.get_left())},{rec(node.get_right())}):{node.dist:.4f}"
        return rec(tree)
    except Exception as e:
        return f"Erreur NJ : {e}"


@cache_data
def detect_roh(gt, snp_df_json, min_snps=30, min_kb=500.0):
    snp_df = pd.read_json(snp_df_json)
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    n_ind = gt.shape[0]
    rohs, froh_bp = [], np.zeros(n_ind)
    genome_bp_total = 0.0

    for chrom in pd.unique(chr_arr):
        idxs = np.where(chr_arr == chrom)[0]
        if len(idxs) < min_snps:
            continue
        order = np.argsort(bp[idxs]); idxs = idxs[order]
        chrom_span = bp[idxs[-1]] - bp[idxs[0]]
        if chrom_span < 1:
            continue
        genome_bp_total += chrom_span

        for i in range(n_ind):
            col = gt[i, idxs]; start_k = None
            for k in range(len(idxs)):
                is_hom = (col[k] == 0) or (col[k] == 2)
                if is_hom:
                    if start_k is None:
                        start_k = k
                else:
                    if start_k is not None:
                        n_run = k - start_k
                        length_kb = (bp[idxs[k - 1]] - bp[idxs[start_k]]) / 1000.0
                        if n_run >= min_snps and length_kb >= min_kb:
                            rohs.append({"IID_idx": i, "CHR": chrom,
                                         "start_bp": int(bp[idxs[start_k]]),
                                         "end_bp": int(bp[idxs[k - 1]]),
                                         "n_snp": int(n_run),
                                         "length_kb": float(length_kb)})
                            froh_bp[i] += bp[idxs[k - 1]] - bp[idxs[start_k]]
                        start_k = None
            if start_k is not None:
                n_run = len(idxs) - start_k
                length_kb = (bp[idxs[-1]] - bp[idxs[start_k]]) / 1000.0
                if n_run >= min_snps and length_kb >= min_kb:
                    rohs.append({"IID_idx": i, "CHR": chrom,
                                 "start_bp": int(bp[idxs[start_k]]),
                                 "end_bp": int(bp[idxs[-1]]),
                                 "n_snp": int(n_run),
                                 "length_kb": float(length_kb)})
                    froh_bp[i] += bp[idxs[-1]] - bp[idxs[start_k]]

    froh = froh_bp / genome_bp_total if genome_bp_total > 0 else froh_bp
    roh_df = (pd.DataFrame(rohs) if rohs else
              pd.DataFrame(columns=["IID_idx", "CHR", "start_bp",
                                    "end_bp", "n_snp", "length_kb"]))
    return roh_df, froh


@cache_data
def selection_signatures(gt, snp_df, pop_labels):
    fst = fst_per_snp(gt, np.asarray(pop_labels))
    hom = np.nanmean((gt == 0) | (gt == 2), axis=0)
    df = snp_df.copy()
    df["FST"] = fst; df["HOM"] = hom
    hom_std = df["HOM"].std()
    df["SCORE"] = (df["FST"].fillna(0) *
                   (df["HOM"] - df["HOM"].mean()) /
                   (hom_std if hom_std > 1e-9 else 1.0))
    return df.sort_values("SCORE", ascending=False)


# ============================================================
# GWAS MODULE (LMM-EMMAX)
# ============================================================

def load_phenotypes_file(file_bytes, filename=""):
    if filename.endswith(".gz"):
        try:
            raw = gzip.decompress(file_bytes)
        except Exception:
            raw = file_bytes
    else:
        raw = file_bytes
    text = raw.decode("utf-8", errors="replace")
    first_line = text.splitlines()[0] if text else ""
    sep = "\t" if first_line.count("\t") > first_line.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def match_phenotypes(ind_df, pheno_df, iid_col=None):
    if iid_col is None:
        for c in ["IID", "id", "ID", "sample", "Sample", "animal", "Animal"]:
            if c in pheno_df.columns:
                iid_col = c
                break
    if iid_col is None:
        raise ValueError(
            f"Aucune colonne IID trouvée. Colonnes dispo : "
            f"{list(pheno_df.columns)}")

    p = pheno_df.copy()
    p[iid_col] = p[iid_col].astype(str)
    ind = ind_df.copy()
    ind["_key"] = ind["IID"].astype(str)
    p["_key"] = p[iid_col]

    merged = ind[["_key", "FID", "IID"]].merge(
        p.drop(columns=[iid_col]), on="_key", how="left")
    merged = merged.drop(columns="_key")
    return merged, iid_col


def _vc_estimate_delta(Y, X0, K):
    n = len(Y)
    d, U = np.linalg.eigh(K)
    d = np.maximum(d, 0)
    Yt = U.T @ Y
    X0t = U.T @ X0

    def neg_loglik(delta):
        if delta <= 0:
            return 1e18
        w = 1.0 / np.sqrt(delta * d + 1.0)
        Yw = Yt * w
        Xw = X0t * w[:, None]
        try:
            beta, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except np.linalg.LinAlgError:
            return 1e18
        resid = Yw - Xw @ beta
        s2 = float(np.sum(resid ** 2) / n)
        if s2 <= 0:
            return 1e18
        return -(-0.5 * (n * np.log(s2) +
                         np.sum(np.log(delta * d + 1.0))))

    grid = np.logspace(-3, 3, 30)
    lls = np.array([neg_loglik(dl) for dl in grid])
    i_best = int(np.argmin(lls))
    lo = grid[max(0, i_best - 1)]
    hi = grid[min(len(grid) - 1, i_best + 1)]
    try:
        res = minimize_scalar(neg_loglik, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-4})
        delta_opt = float(res.x)
    except Exception:
        delta_opt = float(grid[i_best])
    return max(delta_opt, 1e-6)


@cache_data
def run_gwas_lmm(gt, pheno_vec, K, covariates,
                 maf_min=0.05, missing_max=0.05, use_lmm=True):
    n_all, n_snp = gt.shape
    pheno_vec = np.asarray(pheno_vec, dtype=float)

    valid_ind = np.isfinite(pheno_vec)
    if valid_ind.sum() < 10:
        raise ValueError("Moins de 10 individus avec phénotype valide.")

    Y = pheno_vec[valid_ind]
    gt_v = gt[valid_ind]
    K_v = np.asarray(K)[np.ix_(valid_ind, valid_ind)]

    if covariates is None or covariates.size == 0:
        X0 = np.ones((len(Y), 1))
    else:
        cov = np.asarray(covariates)[valid_ind]
        X0 = np.column_stack([np.ones(len(Y)), cov])

    keep_cols = np.ones(X0.shape[1], dtype=bool)
    for j in range(1, X0.shape[1]):
        if X0[:, j].std() < 1e-10:
            keep_cols[j] = False
    X0 = X0[:, keep_cols]

    n = len(Y)

    if use_lmm:
        delta = _vc_estimate_delta(Y, X0, K_v)
        d, U = np.linalg.eigh(K_v)
        d = np.maximum(d, 0)
        w = 1.0 / np.sqrt(delta * d + 1.0)
        Yt = (U.T @ Y) * w
        X0t = (U.T @ X0) * w[:, None]
    else:
        Yt = Y
        X0t = X0

    out = {
        "SNP_idx": [], "MAF": [], "BETA": [], "SE": [], "T": [], "P": [],
    }

    for j in range(n_snp):
        g = gt_v[:, j].astype(float)
        nan_mask = ~np.isfinite(g)
        miss_j = nan_mask.mean()
        if miss_j > missing_max:
            continue

        if nan_mask.any():
            m_g = np.nanmean(g)
            if not np.isfinite(m_g):
                continue
            g = np.where(nan_mask, m_g, g)

        p = g.mean() / 2.0
        maf_j = min(p, 1.0 - p)
        if maf_j < maf_min:
            continue

        if use_lmm:
            gw = (U.T @ g) * w
            X = np.column_stack([X0t, gw])
        else:
            X = np.column_stack([X0t, g])

        try:
            beta, *_ = np.linalg.lstsq(X, Yt, rcond=None)
        except np.linalg.LinAlgError:
            continue

        resid = Yt - X @ beta
        dof = max(n - X.shape[1], 1)
        s2 = float(np.sum(resid ** 2) / dof)
        if s2 <= 0:
            continue

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        se = float(np.sqrt(s2 * XtX_inv[-1, -1]))
        if se <= 0 or not np.isfinite(se):
            continue

        t_val = beta[-1] / se
        p_val = float(2 * t_dist.sf(abs(t_val), df=dof))

        out["SNP_idx"].append(j)
        out["MAF"].append(float(maf_j))
        out["BETA"].append(float(beta[-1]))
        out["SE"].append(se)
        out["T"].append(float(t_val))
        out["P"].append(p_val)

    if not out["SNP_idx"]:
        return pd.DataFrame(
            columns=["SNP_idx", "MAF", "BETA", "SE", "T", "P"])

    return pd.DataFrame(out)


def annotate_gwas_results(res_df, snp_df):
    if res_df.empty:
        return res_df
    idx = res_df["SNP_idx"].astype(int).values
    res_df = res_df.copy()
    res_df["SNP"] = snp_df["SNP"].astype(str).values[idx]
    res_df["CHR"] = snp_df["CHR"].astype(str).values[idx]
    res_df["BP"] = snp_df["BP"].astype(int).values[idx]
    return res_df[["SNP_idx", "SNP", "CHR", "BP", "MAF",
                   "BETA", "SE", "T", "P"]]


def compute_lambda_gc(pvals):
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10:
        return np.nan
    chi2_obs = stats.chi2.isf(p, df=1)
    return float(np.median(chi2_obs) / stats.chi2.ppf(0.5, df=1))


def gwas_thresholds(pvals, alpha=0.05):
    p = np.asarray(pvals)
    p = p[np.isfinite(p)]
    n = len(p)
    if n == 0:
        return {"bonferroni": np.nan, "fdr_05": np.nan, "n_tests": 0}
    bonf = alpha / n
    ps = np.sort(p)
    bh = ps * n / np.arange(1, n + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    sig_idx = np.where(bh <= alpha)[0]
    fdr_thr = float(ps[sig_idx[-1]]) if len(sig_idx) else np.nan
    return {"bonferroni": float(bonf), "fdr_05": fdr_thr, "n_tests": int(n)}


def plot_gwas_manhattan(res_df, bonferroni=None, fdr=None,
                        title="GWAS Manhattan"):
    df = res_df.copy()
    df = df[np.isfinite(df["P"])].reset_index(drop=True)
    if df.empty:
        return None
    df["-log10P"] = -np.log10(np.clip(df["P"], 1e-300, 1))

    order = df["CHR"].map(_chr_sort_key)
    df["_k0"] = order.map(lambda t: t[0])
    df["_k1"] = order.map(lambda t: t[1])
    df["_k2"] = order.map(lambda t: t[2])
    df = df.sort_values(["_k0", "_k1", "BP"]).reset_index(drop=True)

    cum, ticks, labels = 0, [], []
    pos = np.zeros(len(df), dtype=int)
    for chrom in df["CHR"].unique():
        idx = np.where(df["CHR"] == chrom)[0]
        pos[idx] = np.arange(cum, cum + len(idx))
        ticks.append((cum + cum + len(idx) - 1) / 2)
        labels.append(str(chrom))
        cum += len(idx)
    df["x"] = pos

    fig = go.Figure()
    for i, chrom in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == chrom]
        color = "#2c3e50" if i % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["-log10P"], mode="markers",
            marker=dict(size=5, color=color),
            text=[f"{s}<br>CHR{c}:{b}<br>P={p:.2e}<br>β={bb:.3f}"
                  for s, c, b, p, bb in zip(
                      sub["SNP"], sub["CHR"], sub["BP"],
                      sub["P"], sub["BETA"])],
            hoverinfo="text", showlegend=False))

    if bonferroni is not None and np.isfinite(bonferroni):
        fig.add_hline(y=-np.log10(bonferroni), line_dash="dash",
                      line_color="red",
                      annotation_text=f"Bonferroni {bonferroni:.2e}",
                      annotation_position="top right")
    if fdr is not None and np.isfinite(fdr):
        fig.add_hline(y=-np.log10(fdr), line_dash="dot",
                      line_color="orange",
                      annotation_text=f"FDR 5% {fdr:.2e}",
                      annotation_position="bottom right")

    fig.update_layout(title=title, xaxis_title="Chromosome",
                      yaxis_title="-log10(P)", height=520,
                      margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_gwas_qq(pvals, lambda_gc=None):
    p = np.asarray(pvals)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10:
        return None
    p = np.sort(p)
    n = len(p)
    expected = (np.arange(1, n + 1) - 0.5) / n
    obs = -np.log10(p)
    exp = -np.log10(expected)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exp, y=obs, mode="markers",
                             marker=dict(size=5, color="#2c3e50"),
                             name="Observé"))
    lim = max(exp.max(), obs.max()) * 1.05
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                             line=dict(color="red", dash="dash"),
                             name="Attendu (H0)"))
    sub = (f"λGC = {lambda_gc:.3f}"
           if lambda_gc is not None and np.isfinite(lambda_gc) else "")
    fig.update_layout(title=f"QQ plot {sub}",
                      xaxis_title="Attendu -log10(P)",
                      yaxis_title="Observé -log10(P)",
                      height=520,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_locuszoom(res_df, snp_df, chrom, center_bp, window_kb=500):
    chrom = str(chrom)
    window_bp = window_kb * 1000
    df = res_df.copy()
    df = df[df["CHR"].astype(str) == chrom]
    df = df[(df["BP"] >= center_bp - window_bp) &
            (df["BP"] <= center_bp + window_bp)]
    if df.empty:
        return None
    df["-log10P"] = -np.log10(np.clip(df["P"], 1e-300, 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["BP"] / 1e6, y=df["-log10P"], mode="markers",
        marker=dict(size=8, color=df["-log10P"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="-log10P")),
        text=[f"{s}<br>P={p:.2e}<br>β={b:.3f}"
              for s, p, b in zip(df["SNP"], df["P"], df["BETA"])],
        hoverinfo="text"))
    fig.add_vline(x=center_bp / 1e6, line_dash="dash", line_color="red",
                  annotation_text="Lead SNP",
                  annotation_position="top")
    fig.update_layout(
        title=f"LocusZoom — CHR {chrom} : "
              f"{center_bp/1e6:.2f} Mb ± {window_kb} kb",
        xaxis_title="Position (Mb)",
        yaxis_title="-log10(P)", height=500)
    return fig


# ============================================================
# INTERPRÉTATION IA
# ============================================================

def interpret_qc(qc_stats):
    pct_ind = qc_stats["excluded_ind"] / max(qc_stats["n_ind_init"], 1) * 100
    pct_snp = qc_stats["excluded_snp"] / max(qc_stats["n_snp_init"], 1) * 100
    if pct_ind < 2 and pct_snp < 10:
        v, e = "🟢 EXCELLENTE", "Qualité de génotypage très élevée."
    elif pct_ind < 5 and pct_snp < 25:
        v, e = "🟡 ACCEPTABLE", "Qualité correcte, artéfacts éliminés normalement."
    else:
        v, e = "🔴 FAIBLE", "Exclusion élevée → puce/ADN problématique."
    return {"verdict": v, "explanation": e,
            "metrics": {"ind_exclus_%": round(pct_ind, 2),
                        "snp_exclus_%": round(pct_snp, 2)},
            "recommendation": ("Passez à l'étape suivante."
                               if "🟢" in v else "Vérifiez la qualité des puces.")}


def interpret_maf(maf_values):
    m = np.asarray(maf_values)
    m = m[np.isfinite(m)]
    mm = float(m.mean()); pct = float((m < 0.05).mean() * 100)
    if mm > 0.25:
        v, e = "🟢 Spectre riche", "Bonne diversité, variants informatifs."
    elif mm > 0.15:
        v, e = "🟡 Diversité modérée", "Population avec diversité moyenne."
    else:
        v, e = "🔴 Diversité faible", "Population consanguine ou échantillon réduit."
    return {"verdict": v, "explanation": e,
            "metrics": {"MAF_moyen": round(mm, 4), "pct_MAF<0.05": round(pct, 2)},
            "recommendation": f"Filtre MAF 0.05 élimine {pct:.1f}% des SNPs."}


def interpret_fst(fst_values):
    m = float(np.nanmean(np.asarray(fst_values)))
    if m < 0.05:
        v, e = "🟢 Faible", "Fort flux génique."
    elif m < 0.15:
        v, e = "🟡 MODÉRÉE", "Races distinctes, échanges historiques."
    elif m < 0.25:
        v, e = "🟠 FORTE", "Races isolées génétiquement."
    else:
        v, e = "🔴 TRÈS FORTE", "Quasi-espèces distinctes."
    return {"verdict": v, "explanation": e,
            "metrics": {"FST_moyen": round(m, 4)},
            "recommendation": "Utiliser PC1-PC10 + MDS comme covariables GWAS."}


def interpret_admixture(cv_results):
    ks = sorted(cv_results.keys())
    errors = [cv_results[k]["mean"] for k in ks]
    drops = [errors[i] - errors[i + 1] for i in range(len(errors) - 1)]
    k_opt = ks[int(np.argmax(drops)) + 1] if drops else ks[0]
    return {"verdict": f"🟢 K optimal = {k_opt}",
            "explanation": "Nombre d'ancestralités minimisant la cross-entropy.",
            "metrics": {f"K={k}": round(cv_results[k]["mean"], 4) for k in ks},
            "recommendation": f"Relancer l'admixture avec K={k_opt}."}


def interpret_roh(froh):
    m = float(np.nanmean(np.asarray(froh)))
    if m < 0.02:
        v, e = "🟢 Consanguinité faible", "Bonne diversité."
    elif m < 0.10:
        v, e = "🟡 Consanguinité modérée", "Niveau normal."
    else:
        v, e = "🔴 Consanguinité élevée", "Risque de dépression."
    return {"verdict": v, "explanation": e,
            "metrics": {"FROH_moyen": round(m, 4),
                        "FROH_max": round(float(np.nanmax(froh)), 4)},
            "recommendation": ("Croiser avec lignées extérieures"
                               if m > 0.10 else "Population saine.")}


def interpret_with_llm(context, api_key=None, model="llama-3.3-70b-versatile"):
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ Pas de clé API Groq — interprétation heuristique affichée."
    try:
        import requests
        prompt = f"""Tu es un généticien expert en bioinformatique bovine.
Analyse ces résultats : (1) Synthèse, (2) Interprétation biologique,
(3) Points d'alerte, (4) Recommandations. Français, ≤ 400 mots.

{json.dumps(context, indent=2, default=str)[:6000]}"""
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model, "messages": [
                {"role": "system", "content": "Expert génétique bovine."},
                {"role": "user", "content": prompt}],
                "temperature": 0.3, "max_tokens": 1200},
            timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Erreur LLM : {e}"


def render_ai_panel(module, data, key_suffix=""):
    st.divider()
    st.markdown(f"### 🤖 Interprétation IA — {module}")
    interp_map = {"QC": interpret_qc, "MAF": interpret_maf,
                  "FST": interpret_fst, "Admixture": interpret_admixture,
                  "ROH": interpret_roh}
    if module in interp_map:
        result = interp_map[module](data)
        st.success(f"**{result['verdict']}** — {result['explanation']}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 Métriques**"); st.json(result["metrics"])
        with c2:
            st.markdown("**✅ Recommandation**")
            st.info(result["recommendation"])

    with st.expander("🧠 Analyse LLM approfondie (Groq gratuit)"):
        st.caption("Clé gratuite sur https://console.groq.com/keys")
        k = st.text_input("Clé API Groq", type="password",
                          key=f"groq_{module}_{key_suffix}")
        if st.button(f"🚀 Analyser {module}",
                     key=f"btn_llm_{module}_{key_suffix}"):
            ctx = (data if isinstance(data, dict) else {"raw": str(data)[:2000]})
            with st.spinner("Analyse LLM..."):
                st.markdown(interpret_with_llm(ctx, api_key=k))


# ============================================================
# EXPORT
# ============================================================

def _dosage_to_plink_bits(col):
    n = len(col); bits = np.zeros(n, dtype=np.uint8)
    bits[np.isnan(col)] = 0b01
    bits[col == 0] = 0b11; bits[col == 1] = 0b10; bits[col == 2] = 0b00
    return bits


def build_plink_bed(gt):
    n_ind, n_snp = gt.shape
    n_bytes = (n_ind + 3) // 4
    buf = bytearray([0x6C, 0x1B, 0x01])
    for j in range(n_snp):
        bits = _dosage_to_plink_bits(gt[:, j])
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for k in range(n_ind):
            packed[k // 4] |= (bits[k] & 0x03) << (2 * (k % 4))
        buf.extend(packed.tobytes())
    return bytes(buf)


def build_plink_bim(snp_df):
    lines = []
    has_a1 = "A1" in snp_df.columns
    has_a2 = "A2" in snp_df.columns
    for _, r in snp_df.iterrows():
        a1 = str(r["A1"]) if has_a1 else "A"
        a2 = str(r["A2"]) if has_a2 else "G"
        if a1 in ("nan", ""): a1 = "A"
        if a2 in ("nan", "", a1): a2 = "G" if a1 == "A" else "A"
        lines.append(f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t"
                     f"{int(r['BP'])}\t{a1}\t{a2}")
    return "\n".join(lines) + "\n"


def build_plink_fam(ind_df):
    return "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9"
                     for _, r in ind_df.iterrows()) + "\n"


def build_vcf_output(gt, ind_df, snp_df, project="BovineSNP"):
    n_ind, n_snp = gt.shape
    has_a1 = "A1" in snp_df.columns
    has_a2 = "A2" in snp_df.columns
    header = ["##fileformat=VCFv4.2",
              f"##source=BovineSNPPlatform-{project}",
              f"##fileDate={datetime.now():%Y%m%d}",
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">']
    samples = ind_df["IID"].astype(str).tolist()
    header.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                  + "\t".join(samples))
    body = []
    for j in range(n_snp):
        r = snp_df.iloc[j]
        ref = str(r["A1"]) if has_a1 else "A"
        alt = str(r["A2"]) if has_a2 else "G"
        if ref in ("nan", ""): ref = "A"
        if alt in ("nan", "", ref): alt = "G" if ref == "A" else "A"
        gts = []
        for i in range(n_ind):
            v = gt[i, j]
            gts.append("./." if np.isnan(v)
                       else "0/0" if v == 0
                       else "0/1" if v == 1 else "1/1")
        body.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t{ref}\t{alt}"
                    f"\t.\tPASS\t.\tGT\t" + "\t".join(gts))
    return "\n".join(header + body) + "\n"


def build_plink_zip(gt, ind_df, snp_df, prefix="bovine_qc"):
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed", build_plink_bed(gt))
        zf.writestr(f"{prefix}.bim", build_plink_bim(snp_df))
        zf.writestr(f"{prefix}.fam", build_plink_fam(ind_df))
    return bio.getvalue()


# ============================================================
# VISUALISATION
# ============================================================

def plot_hist(values, title, xlabel, color="#3498db"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=80, marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Fréquence",
                      height=380, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def plot_missingness_dashboard(miss_ind, miss_snp):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Missingness / individu",
                                        "Missingness / SNP"))
    fig.add_trace(go.Histogram(x=miss_ind, nbinsx=60,
                               marker_color="skyblue"), row=1, col=1)
    fig.add_trace(go.Histogram(x=miss_snp, nbinsx=60,
                               marker_color="coral"), row=1, col=2)
    fig.update_layout(height=400, showlegend=False,
                      margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=1)
    fig.update_xaxes(title_text="Fréquence manquante", row=1, col=2)
    fig.update_yaxes(title_text="Nombre", row=1, col=1)
    return fig


def plot_pca(scores, var_pct, labels):
    df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="PC1", y="PC2", color="Population",
                     title=f"PCA — PC1 ({var_pct[0]:.1f}%) vs "
                           f"PC2 ({var_pct[1]:.1f}%)", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig


def plot_mds(coords, labels):
    df = pd.DataFrame({"MDS1": coords[:, 0], "MDS2": coords[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Population",
                     title="MDS (IBS) — Structure", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig


def plot_manhattan(fst, chr_col, threshold_q=0.999):
    df = pd.DataFrame({"FST": np.asarray(fst), "CHR": np.asarray(chr_col)})
    df = df.dropna(subset=["FST"]).reset_index(drop=True)
    if df.empty:
        return None
    order = df["CHR"].map(_chr_sort_key)
    df["_k0"] = order.map(lambda t: t[0])
    df["_k1"] = order.map(lambda t: t[1])
    df["_k2"] = order.map(lambda t: t[2])
    df = df.sort_values(["_k0", "_k1", "_k2"]).reset_index(drop=True)

    cumulative, ticks, labels = 0, [], []
    cum_pos = np.zeros(len(df), dtype=int)
    for chrom in df["CHR"].unique():
        sub_idx = df.index[df["CHR"] == chrom]
        s, e = cumulative, cumulative + len(sub_idx)
        cum_pos[s:e] = np.arange(s, e)
        ticks.append((s + e) / 2); labels.append(str(chrom))
        cumulative = e
    df["x"] = cum_pos

    fig = go.Figure()
    for idx_chr, chrom in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == chrom]
        color = "#2c3e50" if idx_chr % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["FST"], mode="markers",
                                 marker=dict(size=5, color=color),
                                 showlegend=False, hoverinfo="skip"))
    q_upper = float(np.nanquantile(fst, threshold_q))
    fig.add_hline(y=q_upper, line_dash="dash", line_color="red",
                  annotation_text=f"Top {100*(1-threshold_q):.1f}% = {q_upper:.4f}",
                  annotation_position="top right")
    fig.update_layout(title="Manhattan Plot — FST/SNP",
                      xaxis_title="Chromosome", yaxis_title="FST",
                      height=500, margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_ld_decay(ld_df, bin_kb=20):
    if ld_df is None or ld_df.empty:
        return None
    d = ld_df.copy(); d["bin"] = (d["dist_kb"] // bin_kb) * bin_kb
    agg = d.groupby("bin")["r2"].mean().reset_index()
    fig = px.line(agg, x="bin", y="r2",
                  labels={"bin": "Distance (kb)", "r2": "r² moyen"},
                  title="LD decay", height=450)
    fig.update_traces(line=dict(color="royalblue", width=3))
    return fig


def plot_kinship_heatmap(G, labels):
    labels = [str(x) for x in labels]
    n = len(labels)
    seen, uniq = {}, []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        uniq.append(lab if seen[lab] == 1 else f"{lab}#{seen[lab]}")
    vmin = float(np.nanpercentile(G, 1)); vmax = float(np.nanpercentile(G, 99))
    if not np.isfinite(vmin) or vmin == vmax:
        vmin, vmax = -0.3, 0.5
    custom = np.stack([np.repeat(uniq, n).reshape(n, n),
                       np.tile(uniq, n).reshape(n, n)], axis=-1)
    fig = go.Figure(data=go.Heatmap(
        z=G, colorscale="RdBu", zmid=0.0, zmin=vmin, zmax=vmax,
        colorbar=dict(title="Kinship"),
        hovertemplate="%{customdata[0]} × %{customdata[1]}<br>"
                      "φ = %{z:.3f}<extra></extra>",
        customdata=custom))
    fig.update_layout(title="Matrice de parenté (KING robust)",
                      height=650, xaxis=dict(showticklabels=False),
                      yaxis=dict(showticklabels=False, autorange="reversed"))
    return fig


def plot_admixture(Q, labels, pop_labels, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)])
    df["IID"] = labels; df["Pop"] = pop_labels
    df["_dom"] = Q.argmax(1)
    df["_po"] = pd.Categorical(pop_labels, categories=sorted(set(pop_labels)))
    df = df.sort_values(["_po", "_dom"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))
    palette = (px.colors.qualitative.Set2
               + px.colors.qualitative.Set3)[:K]
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(
            x=df["x"], y=df[f"K{k+1}"], name=f"Composante {k+1}",
            marker_color=palette[k % len(palette)],
            hovertemplate="Ind %{customdata}<br>K" + str(k + 1)
                          + " = %{y:.2f}<extra></extra>",
            customdata=df["IID"]))
    fig.update_layout(barmode="stack",
                      title=f"Proportions d'ancestralité (K={K})",
                      xaxis_title="Individus triés",
                      yaxis_title="Proportion", height=500)
    return fig, df


def plot_fst_pairwise(matrix, pops):
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=pops, y=pops, colorscale="Viridis",
        colorbar=dict(title="FST"),
        text=np.round(matrix, 4), texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}<br>FST = %{z:.4f}<extra></extra>"))
    fig.update_layout(title="FST pairwise", height=550)
    return fig


def plot_roh_histogram(froh, labels):
    df = pd.DataFrame({"FROH": froh, "Population": labels})
    return px.histogram(df, x="FROH", color="Population", nbins=50,
                        title="Distribution de FROH", height=450)


def plot_ne_curves(ne_dict):
    fig = go.Figure()
    palette = (px.colors.qualitative.Set2 + px.colors.qualitative.Set3)
    for i, (race, df) in enumerate(ne_dict.items()):
        if df is None or df.empty:
            continue
        fig.add_trace(go.Scatter(x=df["GenAgo"], y=df["Ne"],
                                 mode="lines+markers", name=race,
                                 line=dict(color=palette[i % len(palette)],
                                           width=3)))
    fig.update_layout(title="Évolution historique de la taille efficace (Ne)",
                      xaxis_title="Générations passées",
                      yaxis_title="Ne (interprétation relative)", height=550)
    return fig


def plot_cv_error(cv_results):
    ks = sorted(cv_results.keys())
    means = [cv_results[k]["mean"] for k in ks]
    stds = [cv_results[k]["std"] for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=means, mode="lines+markers",
                             error_y=dict(type="data", array=stds),
                             line=dict(color="crimson", width=3),
                             marker=dict(size=12)))
    k_opt = ks[int(np.argmin(means))]
    fig.add_vline(x=k_opt, line_dash="dash", line_color="green",
                  annotation_text=f"K optimal = {k_opt}",
                  annotation_position="top")
    fig.update_layout(title="Cross-Entropy Error (choix K)",
                      xaxis_title="K (ancestralités)",
                      yaxis_title="CV Error", height=450)
    return fig


# ============================================================
# RAPPORT HTML
# ============================================================

def build_report_html(config, stats, figures=None, interpretations=None):
    figures = figures or {}; interpretations = interpretations or {}
    first, blocks = True, []
    for title, fig in figures.items():
        if fig is None:
            continue
        html = fig.to_html(full_html=False,
                           include_plotlyjs="cdn" if first else False)
        first = False
        blocks.append(f"<h2>{title}</h2>{html}")

    interp_html = ""
    for module, interp in interpretations.items():
        interp_html += f"""
        <div style="background:#eafaf1;padding:12px;border-radius:8px;margin:10px 0;">
          <h3>{module}</h3>
          <p><b>{interp.get('verdict','')}</b> — {interp.get('explanation','')}</p>
          <p><b>Métriques :</b> {interp.get('metrics', {})}</p>
          <p><b>Recommandation :</b> {interp.get('recommendation','')}</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Rapport Bovine SNP Platform</title>
<style>
 body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #222; }}
 h1 {{ color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 8px; }}
 h2 {{ color: #2471a3; margin-top: 28px; }}
 .summary {{ background: #f4f6f7; padding: 16px; border-radius: 8px; }}
</style></head><body>
<h1>🐄 Rapport Bovine SNP Platform</h1>
<p><b>Projet :</b> {config.get('project_name','N/A')} —
   <b>Date :</b> {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="summary"><h2>Résumé exécutif</h2><ul>
 <li>Individus analysés : <b>{stats['n_ind_final']}</b> (sur {stats['n_ind_init']})</li>
 <li>SNPs retenus : <b>{stats['n_snp_final']}</b> (sur {stats['n_snp_init']})</li>
 <li>Individus exclus : <b>{stats['excluded_ind']}</b></li>
 <li>SNPs exclus : <b>{stats['excluded_snp']}</b></li>
</ul></div>
{interp_html}
{''.join(blocks)}
<hr><p style="font-size:0.85em;color:#666">
Rapport généré par Bovine SNP Platform v4.0.</p>
</body></html>
"""


# ============================================================
# STREAMLIT — ÉTAT
# ============================================================

_STATE_KEYS = ["gt", "ind_df", "snp_df",
               "gt_filt", "ind_filt", "snp_filt", "qc_stats",
               "gt_pruned", "snp_pruned",
               "pca_scores", "pca_var", "mds_coords", "king_matrix",
               "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
               "admixture_Q", "admixture_K", "cv_results",
               "roh_df", "froh", "ne_dict", "reynolds_D", "reynolds_pops",
               "selection_df", "run_requested",
               "_plink_zip", "_vcf_str", "_report_html",
               # GWAS
               "pheno_df", "pheno_matched", "gwas_results",
               "gwas_pheno_name", "gwas_lambda",
               # Axiom converter
               "_conv_samples", "_conv_geno", "_conv_n_snp",
               "_conv_ped", "_conv_map"]


def init_state():
    for k in _STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None


def invalidate_downstream():
    for k in ["pca_scores", "pca_var", "mds_coords", "king_matrix",
              "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
              "admixture_Q", "admixture_K", "cv_results",
              "roh_df", "froh", "ne_dict", "reynolds_D", "reynolds_pops",
              "selection_df", "gt_pruned", "snp_pruned",
              "gwas_results", "gwas_lambda"]:
        st.session_state[k] = None


def reset_all_derived():
    for k in (["gt_filt", "ind_filt", "snp_filt", "qc_stats"] +
              ["pca_scores", "pca_var", "mds_coords", "king_matrix",
               "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
               "admixture_Q", "admixture_K", "cv_results",
               "roh_df", "froh", "ne_dict", "reynolds_D", "reynolds_pops",
               "selection_df", "gt_pruned", "snp_pruned",
               "pheno_df", "pheno_matched", "gwas_results",
               "gwas_pheno_name", "gwas_lambda"]):
        st.session_state[k] = None


def has_data():
    return st.session_state.gt is not None


def has_qc():
    return (st.session_state.gt_filt is not None
            and st.session_state.qc_stats is not None
            and st.session_state.ind_filt is not None
            and st.session_state.snp_filt is not None)


def has_pruned():
    return (st.session_state.gt_pruned is not None
            and st.session_state.snp_pruned is not None)


def _fid_array():
    return st.session_state.ind_filt["FID"].astype(str).to_numpy()


# ============================================================
# STREAMLIT — INTERFACE PRINCIPALE
# ============================================================

def main():
    init_state()
    st.title("🐄 Bovine SNP Platform v4.0")
    st.caption("Pipeline complet — QC · LD Pruning · KING · Structure · "
               "Admixture · SNeP · Reynolds · Sélection · GWAS · "
               "Convertisseur Axiom · IA.")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :",
                        ["Demo", "Upload PED/MAP", "Upload VCF",
                         "🔄 Convertir Axiom → PED"], index=0,
                        key="sb_mode_source")

        if mode == "Demo":
            c1, c2 = st.columns(2)
            n_ind = c1.number_input("Individus", 20, 1000, 150, 10,
                                    key="sb_n_ind")
            n_snp = c2.number_input("SNPs", 100, 10000, 800, 100,
                                    key="sb_n_snp")
            n_pop = st.slider("Populations", 2, 10, 4, key="sb_n_pop")
            if st.button("🎲 Générer le jeu de démo",
                         use_container_width=True, key="sb_btn_demo"):
                with st.spinner("Génération..."):
                    gt, ind_df, snp_df = generate_demo_data(
                        int(n_ind), int(n_snp), int(n_pop))
                    ind_df = reconstruct_fid_from_iid(ind_df)
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind_df
                    st.session_state.snp_df = snp_df
                    reset_all_derived()
                st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")

        elif mode == "Upload PED/MAP":
            st.info("Formats : .ped, .ped.gz, .map, .map.gz")
            ped_f = st.file_uploader("Fichier .ped",
                                     type=["ped", "gz", "txt"],
                                     key="sb_ped_file")
            map_f = st.file_uploader("Fichier .map",
                                     type=["map", "gz", "txt"],
                                     key="sb_map_file")
            if ped_f and map_f:
                if st.button("📥 Charger PED + MAP",
                             use_container_width=True, key="sb_btn_ped"):
                    try:
                        with st.spinner("Parsing .map..."):
                            map_df, _ = parse_map(map_f.read())
                        with st.spinner(f"Parsing .ped "
                                        f"({ped_f.size/1024/1024:.1f} Mo)..."):
                            gt, ind_df, _ = parse_ped(ped_f.read(), len(map_df))
                        if len(map_df) != gt.shape[1]:
                            map_df = map_df.iloc[:gt.shape[1]].reset_index(drop=True)
                        ind_df = reconstruct_fid_from_iid(ind_df)
                        st.session_state.gt = gt
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = map_df
                        reset_all_derived()
                        st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")
                    except Exception as e:
                        st.error(f"❌ {e}")

        elif mode == "Upload VCF":
            vcf_f = st.file_uploader("Fichier .vcf",
                                     type=["vcf", "gz", "txt"],
                                     key="sb_vcf_file")
            if vcf_f is not None:
                if st.button("📥 Charger VCF", use_container_width=True,
                             key="sb_btn_vcf"):
                    try:
                        with st.spinner("Parsing VCF..."):
                            gt, ind_df, snp_df, n_skip, n_multi = parse_vcf(
                                vcf_f.read())
                        ind_df = reconstruct_fid_from_iid(ind_df)
                        st.session_state.gt = gt
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = snp_df
                        reset_all_derived()
                        st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} variants")
                    except Exception as e:
                        st.error(f"❌ {e}")

        else:  # 🔄 Convertir Axiom → PED
            st.info("📄 Sélectionne le fichier brut dans la zone principale "
                    "et clique sur **Convertir**.")

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("Missingness SNP (--geno)", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["geno"], 0.01, key="sb_geno")
        mind = st.slider("Missingness individu (--mind)", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["mind"], 0.01, key="sb_mind")
        maf_thr = st.slider("MAF minimal (--maf)", 0.0, 0.5,
                            DEFAULT_THRESHOLDS["maf"], 0.01, key="sb_maf")
        hwe_thr = st.number_input("HWE p-value (--hwe)",
                                  value=DEFAULT_THRESHOLDS["hwe"],
                                  format="%.0e", key="sb_hwe")
        het_sd = st.slider("Hétérozygotie ±σ intra-race", 1.0, 5.0,
                           DEFAULT_THRESHOLDS["het_sd"], 0.1, key="sb_het_sd")
        ld_r2 = st.slider("LD Pruning r² seuil", 0.05, 0.5,
                          DEFAULT_THRESHOLDS["ld_r2"], 0.05, key="sb_ld_r2")
        king_cutoff = st.slider("KING cutoff", 0.1, 0.5,
                                DEFAULT_THRESHOLDS["king_cutoff"], 0.01,
                                key="sb_king")

        st.divider()
        if st.button("🚀 Pipeline complet", type="primary",
                     use_container_width=True, key="sb_btn_pipeline"):
            if not has_data():
                st.error("Chargez d'abord des données.")
            else:
                st.session_state.run_requested = True

        st.divider()
        st.caption("v4.0 — Conforme manuscrit · IA · GWAS · Export PLINK")

    # ---------- MODE CONVERTISSEUR ----------
    if st.session_state.get("sb_mode_source") == "🔄 Convertir Axiom → PED":
        render_axiom_converter_page()
        return

    # ---------- MAIN ----------
    if not has_data():
        st.info("👉 Générez un jeu de démo, importez PED/MAP ou VCF, "
                "ou utilisez le convertisseur Axiom → PED.")
        st.markdown("""
        ### 🧬 Pipeline complet

        | Phase | Module | Sortie |
        |---|---|---|
        | 1 | Nettoyage PED + autosomes | Cattle.ped propre |
        | 2 | QC (missing, MAF, HWE, het) | Cattle_QC_Clean |
        | 3 | LD Pruning | SNPs indépendants |
        | 4 | KING + PCA + MDS + FST | Structure |
        | 5 | Admixture (NMF + CV) | Ancestralités Q |
        | 6 | LD decay · ROH · SNeP | Démographie |
        | 7 | Reynolds · NJ · Sélection | Publication |
        | 8 | **GWAS (LMM-EMMAX)** | Manhattan + QQ + LocusZoom |
        | IA | Interprétation auto | Recommandations |

        ### 🎁 Nouveautés
        - ✅ Filtrage autosomes 1-29
        - ✅ FID reconstruit depuis IID
        - ✅ LD Pruning (50 5 0.2)
        - ✅ KING robust + cutoff 0.354
        - ✅ CV error pour K optimal
        - ✅ Ne historique (SNeP-like)
        - ✅ Reynolds + arbre NJ
        - ✅ Signatures de sélection
        - ✅ **GWAS LMM-EMMAX complet**
        - ✅ **Convertisseur Axiom → PED**
        - ✅ Interprétation IA automatique
        - ✅ Rapport HTML
        """)
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "✂️ LD Pruning",
                    "🧬 Structure", "🎨 Admixture", "📈 Démographie",
                    "🔍 Sélection", "🧪 GWAS", "🌳 Phylogénie",
                    "📤 Export", "📄 Rapport"])

    # ============ TAB 1 : Aperçu ============
    with tabs[0]:
        step_header("0", "Aperçu des données",
                    "Comptage des individus, SNPs, races.",
                    "Vérifier la cohérence avant toute analyse.",
                    "Aperçu visuel du jeu de données.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations (FID)", ind_df["FID"].nunique())

        st.subheader("Individus")
        st.dataframe(ind_df.head(20), use_container_width=True,
                     key="tab0_df_ind")
        st.subheader("SNPs")
        st.dataframe(snp_df.head(5), use_container_width=True,
                     key="tab0_df_snp")

        with st.expander("🔬 Compatibilité ARS-UCD1.2"):
            chroms = sorted(snp_df["CHR"].astype(str).unique(),
                            key=_chr_sort_key)
            cov = []
            for c in chroms:
                key = str(c).upper().replace("CHR", "")
                if key in ARS_UCD12_LENGTHS:
                    mask = snp_df["CHR"].astype(str) == c
                    max_bp = snp_df.loc[mask, "BP"].max()
                    cov.append({"CHR": c, "max_bp": int(max_bp),
                                "ARS_len": ARS_UCD12_LENGTHS[key],
                                "ok": max_bp <= ARS_UCD12_LENGTHS[key]})
            if cov:
                st.dataframe(pd.DataFrame(cov), use_container_width=True,
                             key="tab0_df_ars")

    # ============ TAB 2 : QC ============
    with tabs[1]:
        st.subheader("🧹 Contrôle Qualité (QC)")
        step_header("3", "QC — Missingness, MAF, HWE, Hétérozygotie",
                    "Cascade : --geno 0.05 → --mind 0.05 → --maf 0.05 → "
                    "--hwe 1e-6 → ±3σ hétérozygotie par race.",
                    "Éliminer SNPs et individus de mauvaise qualité.",
                    "Jeu de données propre.")

        if st.button("🧬 Filtrer autosomes 1-29", use_container_width=True,
                     key="qc_btn_auto"):
            try:
                gt_a, snp_a = filter_autosomes(st.session_state.gt,
                                               st.session_state.snp_df)
                st.session_state.gt = gt_a
                st.session_state.snp_df = snp_a
                st.success(f"✅ Autosomes → {gt_a.shape[1]} SNPs")
            except Exception as e:
                st.error(f"❌ {e}")

        if st.button("▶ Lancer le QC complet", type="primary",
                     use_container_width=True, key="qc_btn_run"):
            try:
                with st.spinner("Filtrage en cascade..."):
                    params = {"geno": geno, "mind": mind, "maf": maf_thr,
                              "hwe": hwe_thr, "het_sd": het_sd}
                    gt_f, ind_f, snp_f, qc_stats = apply_qc_filters(
                        st.session_state.gt, st.session_state.ind_df,
                        st.session_state.snp_df, params)
                    st.session_state.gt_filt = gt_f
                    st.session_state.ind_filt = ind_f
                    st.session_state.snp_filt = snp_f
                    st.session_state.qc_stats = qc_stats
                    invalidate_downstream()
                st.success("✅ QC terminé.")
            except Exception as e:
                st.error(f"❌ {e}")

        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus finaux", s["n_ind_final"],
                      delta=-s["excluded_ind"], delta_color="inverse")
            c2.metric("SNPs finaux", s["n_snp_final"],
                      delta=-s["excluded_snp"], delta_color="inverse")
            c3.metric("Exclus (ind)", s["excluded_ind"])
            c4.metric("Exclus (SNPs)", s["excluded_snp"])

            st.markdown("**📊 Cascade de filtrage**")
            st.json(s.get("trace", {}))

            gt_full = st.session_state.gt
            st.plotly_chart(
                plot_missingness_dashboard(
                    missingness_per_ind(gt_full),
                    missingness_per_snp(gt_full)),
                use_container_width=True, key="qc_plot_missingness")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    plot_hist(maf(gt_full), "Spectre MAF",
                              "MAF", "#2ecc71"),
                    use_container_width=True, key="qc_plot_hist_maf")
            with c2:
                st.plotly_chart(
                    plot_hist(heterozygosity(gt_full),
                              "Hétérozygotie observée",
                              "HET", "#9b59b6"),
                    use_container_width=True, key="qc_plot_hist_het")

            render_ai_panel("QC", s, "tab_qc")
            st.divider()
            render_ai_panel("MAF", maf(gt_full), "tab_maf")

    # ============ TAB 3 : LD Pruning ============
    with tabs[2]:
        st.subheader("✂️ LD Pruning (--indep-pairwise 50 5 0.2)")
        step_header("4.5", "Élagage par déséquilibre de liaison",
                    "Retire les SNPs corrélés (r² > 0.2) dans une fenêtre de 50 SNPs.",
                    "PCA et ADMIXTURE supposent des marqueurs indépendants.",
                    "Sous-ensemble de SNPs quasi-indépendants.")

        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            if st.button("▶ Lancer le LD Pruning", type="primary",
                         use_container_width=True, key="ld_btn_run"):
                try:
                    with st.spinner("LD pruning..."):
                        keep = ld_pruning(st.session_state.gt_filt,
                                          window=50, step=5, r2_thr=ld_r2)
                        st.session_state.gt_pruned = (
                            st.session_state.gt_filt[:, keep])
                        st.session_state.snp_pruned = (
                            st.session_state.snp_filt[keep].reset_index(drop=True))
                    st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs "
                               f"indépendants conservés.")
                except Exception as e:
                    st.error(f"❌ {e}")

            if has_pruned():
                c1, c2, c3 = st.columns(3)
                c1.metric("SNPs avant", st.session_state.gt_filt.shape[1])
                c2.metric("SNPs après", st.session_state.gt_pruned.shape[1])
                pct = (100 * (1 - st.session_state.gt_pruned.shape[1] /
                              st.session_state.gt_filt.shape[1]))
                c3.metric("Réduction", f"{pct:.1f}%")

                with st.spinner("LD decay..."):
                    ld_df = ld_decay(st.session_state.gt_pruned,
                                     st.session_state.snp_pruned["BP"].values,
                                     max_kb=1000, max_snp=1000)
                    st.session_state.ld_df = ld_df
                fig = plot_ld_decay(ld_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True,
                                    key="ld_plot_decay_pruning")

    # ============ TAB 4 : Structure ============
    with tabs[3]:
        st.subheader("🧬 Structure des populations")
        step_header("5.3-5.4", "PCA, MDS, KING kinship",
                    "PCA, MDS (IBS), matrice de parenté KING robuste.",
                    "Visualiser la structure génétique et détecter doublons.",
                    "Nuages 2D avec races en couleur + matrice de parenté.")

        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)

            if st.button("▶ PCA + MDS + KING", type="primary",
                         use_container_width=True, key="struct_btn_run"):
                try:
                    with st.spinner("PCA..."):
                        s_, v_ = pca_analysis(gt_use, 10)
                        st.session_state.pca_scores = s_
                        st.session_state.pca_var = v_
                    with st.spinner("MDS..."):
                        st.session_state.mds_coords = mds_analysis(gt_use, 5)
                    with st.spinner("KING kinship..."):
                        st.session_state.king_matrix = king_kinship(gt_use)
                    st.success("✅ Terminé.")
                except Exception as e:
                    st.error(f"❌ {e}")

            labels = _fid_array()

            if st.session_state.pca_scores is not None:
                st.plotly_chart(plot_pca(st.session_state.pca_scores,
                                         st.session_state.pca_var, labels),
                                use_container_width=True,
                                key="struct_plot_pca")
            if st.session_state.mds_coords is not None:
                st.plotly_chart(plot_mds(st.session_state.mds_coords, labels),
                                use_container_width=True,
                                key="struct_plot_mds")

            if st.session_state.king_matrix is not None:
                K = st.session_state.king_matrix
                keep_mask, n_removed = apply_king_cutoff(
                    K, st.session_state.ind_filt, king_cutoff)
                st.info(f"👨‍👩‍👧 KING cutoff {king_cutoff} → **{n_removed}** "
                        f"individus trop apparentés identifiés.")

                id_labels = (st.session_state.ind_filt["FID"].astype(str) + "_" +
                             st.session_state.ind_filt["IID"].astype(str)).values
                st.plotly_chart(plot_kinship_heatmap(K, id_labels),
                                use_container_width=True,
                                key="struct_plot_king")

                if st.button("🗑️ Appliquer le filtre KING",
                             use_container_width=True, key="struct_btn_king_apply"):
                    st.session_state.ind_filt = (
                        st.session_state.ind_filt[keep_mask].reset_index(drop=True))
                    st.session_state.gt_filt = st.session_state.gt_filt[keep_mask]
                    if has_pruned():
                        st.session_state.gt_pruned = (
                            st.session_state.gt_pruned[keep_mask])
                    st.session_state.king_matrix = None
                    st.success(f"✅ {n_removed} individus retirés.")

    # ============ TAB 5 : Admixture ============
    with tabs[4]:
        st.subheader("🎨 Admixture (NMF + CV error)")
        step_header("6.4", "Modélisation du métissage",
                    "NMF + Cross-Validation pour choisir K optimal.",
                    "Identifier le nombre d'ancestralités sans sur-ajuster.",
                    "Barplots empilés des proportions d'ancestralités.")

        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)

            c1, c2 = st.columns(2)
            K_max = c1.slider("K max pour CV", 3, 10, 6, key="admix_K_max")
            n_reps = c2.slider("Répétitions CV", 1, 5, 3, key="admix_n_reps")

            if st.button("▶ CV error (K optimal)",
                         use_container_width=True, key="admix_btn_cv"):
                try:
                    with st.spinner(f"CV error K=2..{K_max}..."):
                        cv = admix_cv_error(gt_use,
                                            K_range=(2, K_max + 1),
                                            n_reps=int(n_reps))
                        st.session_state.cv_results = cv
                    st.success("✅ CV terminée.")
                except Exception as e:
                    st.error(f"❌ {e}")

            if st.session_state.cv_results is not None:
                fig = plot_cv_error(st.session_state.cv_results)
                st.plotly_chart(fig, use_container_width=True,
                                key="admix_plot_cv_error")
                render_ai_panel("Admixture", st.session_state.cv_results,
                                "tab_admix_cv")

            K = st.slider("Nombre d'ancestralités K", 2, 10, 4, key="admix_K")
            if st.button("▶ Calculer l'admixture",
                         use_container_width=True, key="admix_btn_run"):
                try:
                    with st.spinner(f"NMF K={K}..."):
                        Q, _ = admixture_nmf(gt_use, K=int(K))
                        st.session_state.admixture_Q = Q
                        st.session_state.admixture_K = K
                    st.success(f"✅ Q : {Q.shape}")
                except Exception as e:
                    st.error(f"❌ {e}")

            if st.session_state.admixture_Q is not None:
                Q = st.session_state.admixture_Q
                K = st.session_state.admixture_K
                pops = _fid_array()
                iids = st.session_state.ind_filt["IID"].astype(str).to_numpy()
                fig, df_s = plot_admixture(Q, iids, pops, K)
                st.plotly_chart(fig, use_container_width=True,
                                key="admix_plot_barplot")
                st.subheader("Proportions moyennes par race")
                st.dataframe(df_s.groupby("Pop")[
                    [f"K{k+1}" for k in range(K)]].mean().round(3),
                    use_container_width=True, key="admix_df_means")

    # ============ TAB 6 : Démographie ============
    with tabs[5]:
        st.subheader("📈 Démographie — LD decay, ROH, Ne")
        step_header("6.1, 6.5", "LD decay, ROH, SNeP",
                    "LD en fonction de la distance, ROH, Ne historique.",
                    "Reconstituer l'histoire démographique des races.",
                    "Courbes LD decay, histogramme FROH, courbes Ne.")

        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            sub1, sub2, sub3 = st.tabs(["LD decay", "ROH", "Ne (SNeP-like)"])

            with sub1:
                if st.button("▶ LD decay", key="demo_btn_ld",
                             use_container_width=True):
                    try:
                        with st.spinner("LD decay..."):
                            st.session_state.ld_df = ld_decay(
                                gt_use,
                                st.session_state.snp_filt["BP"].values,
                                max_kb=1000, max_snp=1000)
                        st.success(f"✅ {len(st.session_state.ld_df):,} paires")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if (st.session_state.ld_df is not None
                        and not st.session_state.ld_df.empty):
                    st.plotly_chart(
                        plot_ld_decay(st.session_state.ld_df),
                        use_container_width=True,
                        key="demo_plot_ld_decay")

            with sub2:
                c1, c2 = st.columns(2)
                min_snps_roh = c1.slider("Min SNPs par ROH", 5, 200, 30, 5,
                                         key="demo_min_snps_roh")
                min_kb_roh = c2.slider("Longueur min (kb)", 50, 5000, 500, 50,
                                       key="demo_min_kb_roh")
                if st.button("▶ Détecter les ROH", key="demo_btn_roh",
                             use_container_width=True):
                    try:
                        with st.spinner("Détection ROH..."):
                            snp_json = st.session_state.snp_filt[
                                ["CHR", "BP"]].to_json()
                            roh_df, froh = detect_roh(
                                gt_use, snp_json,
                                min_snps=int(min_snps_roh),
                                min_kb=float(min_kb_roh))
                            st.session_state.roh_df = roh_df
                            st.session_state.froh = froh
                        st.success(f"✅ {len(roh_df)} ROH")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.froh is not None:
                    froh = st.session_state.froh
                    c1, c2, c3 = st.columns(3)
                    c1.metric("FROH moyen", f"{np.mean(froh):.4f}")
                    c2.metric("FROH médian", f"{np.median(froh):.4f}")
                    c3.metric("Total ROH", len(st.session_state.roh_df))
                    labels = _fid_array()
                    st.plotly_chart(
                        plot_roh_histogram(froh, labels),
                        use_container_width=True,
                        key="demo_plot_roh_hist")
                    render_ai_panel("ROH", froh, "tab_roh")

            with sub3:
                st.markdown("**SNeP-like** : Ne historique par race "
                            "(interprétation relative).")
                if st.button("▶ Estimer Ne par race", key="demo_btn_ne",
                             use_container_width=True):
                    try:
                        with st.spinner("Estimation Ne..."):
                            ne_dict = {}
                            pops = st.session_state.ind_filt["FID"].unique()
                            for fid in pops:
                                mask = (st.session_state.ind_filt["FID"]
                                        == fid).values
                                sub = gt_use[mask]
                                if sub.shape[0] < 5:
                                    continue
                                ne_df = estimate_ne_historical(
                                    sub, st.session_state.snp_filt)
                                ne_dict[str(fid)] = ne_df
                            st.session_state.ne_dict = ne_dict
                        st.success(f"✅ Ne estimé pour {len(ne_dict)} races.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.ne_dict:
                    st.plotly_chart(
                        plot_ne_curves(st.session_state.ne_dict),
                        use_container_width=True,
                        key="demo_plot_ne_curves")
                    st.warning("⚠️ Interprétation RELATIVE uniquement.")

    # ============ TAB 7 : Sélection ============
    with tabs[6]:
        st.subheader("🔍 Signatures de sélection")
        step_header("5.2 étendu", "FST/SNP, Manhattan, sélection",
                    "FST par SNP + FST pairwise + score combiné FST × homosité.",
                    "Identifier les régions sous sélection.",
                    "Manhattan plot, matrice FST, top régions.")

        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            sub1, sub2, sub3 = st.tabs(["FST/SNP + Manhattan",
                                         "FST pairwise",
                                         "Signatures de sélection"])

            with sub1:
                threshold_q = st.slider("Quantile outliers", 0.95, 0.9999,
                                        0.999, 0.0001, format="%.4f",
                                        key="sel_threshold_q")
                if st.button("▶ FST par SNP", key="sel_btn_fst",
                             use_container_width=True):
                    try:
                        with st.spinner("FST..."):
                            st.session_state.fst = fst_per_snp(
                                gt_use, _fid_array())
                        st.success("✅ FST calculé.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.fst is not None:
                    fst_c = st.session_state.fst[
                        np.isfinite(st.session_state.fst)]
                    if len(fst_c) > 0:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("FST moyen", f"{fst_c.mean():.4f}")
                        c2.metric("FST médian", f"{np.median(fst_c):.4f}")
                        c3.metric("Top outliers",
                                  f"{np.quantile(fst_c, threshold_q):.4f}")
                        fig = plot_manhattan(
                            st.session_state.fst,
                            st.session_state.snp_filt["CHR"].values,
                            threshold_q=threshold_q)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True,
                                            key="sel_plot_manhattan")
                        render_ai_panel("FST", fst_c, "tab_fst")

            with sub2:
                if st.button("▶ FST pairwise", key="sel_btn_fstp",
                             use_container_width=True):
                    try:
                        with st.spinner("FST pairwise..."):
                            mat, pops = fst_pairwise(gt_use, _fid_array())
                            st.session_state.fst_pairwise_matrix = mat
                            st.session_state.fst_pops = pops
                        st.success("✅ Matrice calculée.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.fst_pairwise_matrix is not None:
                    st.plotly_chart(
                        plot_fst_pairwise(
                            st.session_state.fst_pairwise_matrix,
                            st.session_state.fst_pops),
                        use_container_width=True,
                        key="sel_plot_fst_pairwise")

            with sub3:
                if st.button("▶ Détecter signatures",
                             key="sel_btn_signatures",
                             use_container_width=True):
                    try:
                        with st.spinner("Analyse FST + homosité..."):
                            sel_df = selection_signatures(
                                gt_use, st.session_state.snp_filt,
                                _fid_array())
                            st.session_state.selection_df = sel_df
                        st.success("✅ Signatures détectées.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.selection_df is not None:
                    st.subheader("Top 20 régions sous sélection")
                    st.dataframe(
                        st.session_state.selection_df.head(20),
                        use_container_width=True, key="sel_df_top20")

    # ============ TAB 8 : GWAS ============
    with tabs[7]:
        st.subheader("🧪 GWAS — Association génotype ↔ phénotype")
        step_header("Extension", "GWAS avec LMM (EMMAX-like)",
                    "Association SNP par SNP avec un phénotype quantitatif, "
                    "en contrôlant la structure (PCs) et la parenté (K).",
                    "Identifier les variants associés à un caractère.",
                    "Manhattan plot, QQ plot, λGC, top hits, LocusZoom.")

        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            snp_use = (st.session_state.snp_pruned if has_pruned()
                       else st.session_state.snp_filt)

            st.markdown("### 1️⃣ Charger les phénotypes (CSV/TSV)")
            st.caption("Format attendu : colonne `IID` + colonnes numériques.")

            pheno_file = st.file_uploader(
                "Fichier phénotypes (CSV / TSV / TXT / .gz)",
                type=["csv", "tsv", "txt", "gz"], key="gwas_pheno_file")

            col_a, col_b = st.columns(2)
            with col_a:
                if pheno_file is not None:
                    if st.button("📥 Charger phénotypes",
                                 key="gwas_btn_load_pheno",
                                 use_container_width=True):
                        try:
                            raw = pheno_file.read()
                            df_p = load_phenotypes_file(raw, pheno_file.name)
                            merged, iid_col = match_phenotypes(
                                st.session_state.ind_filt, df_p)
                            st.session_state.pheno_df = df_p
                            st.session_state.pheno_matched = merged
                            st.success(
                                f"✅ {len(df_p)} lignes, colonne ID : "
                                f"`{iid_col}`.")
                        except Exception as e:
                            st.error(f"❌ {e}")

            with col_b:
                if st.button("🎲 Simuler un phénotype démo (PC1 + bruit)",
                             key="gwas_btn_demo_pheno",
                             use_container_width=True):
                    ind = st.session_state.ind_filt
                    n = len(ind)
                    rng = np.random.default_rng(42)
                    if (st.session_state.pca_scores is not None
                            and len(st.session_state.pca_scores) == n):
                        trait = (st.session_state.pca_scores[:, 0] * 2.0
                                 + rng.normal(0, 0.5, n))
                    else:
                        trait = rng.normal(0, 1, n)
                    merged = pd.DataFrame({
                        "FID": ind["FID"].astype(str).values,
                        "IID": ind["IID"].astype(str).values,
                        "trait_simu": trait})
                    st.session_state.pheno_matched = merged
                    st.success("✅ Phénotype `trait_simu` créé.")

            if st.session_state.pheno_matched is None:
                st.info("👉 Chargez un fichier ou cliquez sur "
                        "« Simuler un phénotype démo ».")
            else:
                merged = st.session_state.pheno_matched
                st.dataframe(merged.head(10), use_container_width=True,
                             key="gwas_df_pheno")

                num_cols = [c for c in merged.columns
                            if c not in ("FID", "IID")
                            and pd.api.types.is_numeric_dtype(merged[c])]
                if not num_cols:
                    st.error("Aucune colonne numérique dans les phénotypes.")
                else:
                    st.markdown("### 2️⃣ Paramètres du modèle")
                    c1, c2, c3 = st.columns(3)
                    pheno_col = c1.selectbox("Phénotype", num_cols,
                                             key="gwas_pheno_col")
                    n_pcs = c2.slider("PCs comme covariables", 0, 10, 5,
                                      key="gwas_n_pcs")
                    maf_thr_gwas = c3.slider("MAF min", 0.0, 0.2, 0.05,
                                             0.01, key="gwas_maf")
                    use_lmm = st.checkbox(
                        "🧬 LMM (recommandé — contrôle la parenté)",
                        value=True, key="gwas_use_lmm")

                    covs = np.empty((len(merged), 0))
                    if n_pcs > 0 and st.session_state.pca_scores is not None:
                        pcs = st.session_state.pca_scores
                        if pcs.shape[0] == len(merged):
                            covs = pcs[:, :n_pcs]
                        else:
                            st.warning("⚠️ Dimensions PCA ≠ individus → "
                                       "PCs ignorées.")

                    st.markdown("### 3️⃣ Lancer l'analyse")
                    if st.button("🚀 Lancer GWAS", type="primary",
                                 use_container_width=True, key="gwas_btn_run"):
                        try:
                            pheno_vec = merged[pheno_col].to_numpy(dtype=float)
                            K_use = (st.session_state.king_matrix
                                     if st.session_state.king_matrix is not None
                                     else np.eye(len(merged)))
                            with st.spinner("Estimation variance + GWAS..."):
                                res = run_gwas_lmm(
                                    gt_use, pheno_vec, K_use, covs,
                                    maf_min=float(maf_thr_gwas),
                                    use_lmm=bool(use_lmm))
                                res = annotate_gwas_results(res, snp_use)
                                st.session_state.gwas_results = res
                                st.session_state.gwas_pheno_name = pheno_col
                                st.session_state.gwas_lambda = \
                                    compute_lambda_gc(res["P"].values)
                            st.success(f"✅ {len(res)} SNPs testés.")
                        except Exception as e:
                            st.error(f"❌ {e}")

                    if st.session_state.gwas_results is not None:
                        res = st.session_state.gwas_results
                        if res.empty:
                            st.warning("Aucun SNP testé (filtres trop stricts ?).")
                        else:
                            thr = gwas_thresholds(res["P"].values)
                            lam = st.session_state.gwas_lambda

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("SNPs testés", len(res))
                            c2.metric("λGC",
                                      f"{lam:.3f}"
                                      if np.isfinite(lam) else "N/A")
                            c3.metric("Seuil Bonferroni",
                                      f"{thr['bonferroni']:.2e}")
                            n_sig = int((res["P"] <= thr["bonferroni"]).sum())
                            c4.metric("Hits Bonferroni", n_sig)

                            if np.isfinite(lam) and lam > 1.1:
                                st.warning(
                                    f"⚠️ λGC = {lam:.3f} > 1.1 → "
                                    "structure résiduelle.")

                            fig_m = plot_gwas_manhattan(
                                res, bonferroni=thr["bonferroni"],
                                fdr=thr.get("fdr_05"),
                                title=f"Manhattan GWAS — "
                                      f"{st.session_state.gwas_pheno_name}")
                            if fig_m is not None:
                                st.plotly_chart(fig_m,
                                                use_container_width=True,
                                                key="gwas_plot_manhattan")

                            fig_qq = plot_gwas_qq(res["P"].values, lam)
                            if fig_qq is not None:
                                st.plotly_chart(fig_qq,
                                                use_container_width=True,
                                                key="gwas_plot_qq")

                            st.subheader("🏆 Top 20 SNPs")
                            top = res.sort_values("P").head(20).copy()
                            top["-log10P"] = -np.log10(
                                np.clip(top["P"], 1e-300, 1))
                            st.dataframe(top, use_container_width=True,
                                         key="gwas_df_top")

                            st.subheader("🔍 Zoom régional (LocusZoom)")
                            if not top.empty:
                                lead = st.selectbox(
                                    "SNP d'intérêt",
                                    top["SNP"].astype(str).tolist(),
                                    key="gwas_lead_snp")
                                row = top[top["SNP"].astype(str) == lead].iloc[0]
                                window_kb = st.slider(
                                    "Fenêtre (kb)", 50, 2000, 500, 50,
                                    key="gwas_window_kb")
                                fig_lz = plot_locuszoom(
                                    res, snp_use, row["CHR"],
                                    int(row["BP"]), window_kb=window_kb)
                                if fig_lz is not None:
                                    st.plotly_chart(
                                        fig_lz, use_container_width=True,
                                        key="gwas_plot_locuszoom")

                            st.subheader("💾 Export")
                            csv = res.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "⬇ Télécharger résultats GWAS (CSV)",
                                data=csv,
                                file_name=(f"gwas_"
                                           f"{st.session_state.gwas_pheno_name}"
                                           f".csv"),
                                mime="text/csv",
                                key="gwas_dl_results",
                                use_container_width=True)

    # ============ TAB 9 : Phylogénie ============
    with tabs[8]:
        st.subheader("🌳 Phylogénie — Reynolds + Neighbor-Joining")
        step_header("6.7 étendu", "Distance de Reynolds + NJ",
                    "Calcul Reynolds + arbre NJ (Newick).",
                    "Visualiser les relations évolutives entre races.",
                    "Matrice + arbre phylogénétique.")

        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            if st.button("▶ Reynolds + NJ", type="primary",
                         use_container_width=True, key="phylo_btn_run"):
                try:
                    with st.spinner("Calcul Reynolds..."):
                        D, pops = reynolds_distance(gt_use, _fid_array())
                        st.session_state.reynolds_D = D
                        st.session_state.reynolds_pops = pops
                    st.success("✅ Matrice Reynolds calculée.")
                except Exception as e:
                    st.error(f"❌ {e}")

            if st.session_state.reynolds_D is not None:
                st.subheader("Matrice de distance (Reynolds)")
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.reynolds_D,
                    x=st.session_state.reynolds_pops,
                    y=st.session_state.reynolds_pops,
                    colorscale="Blues",
                    text=np.round(st.session_state.reynolds_D, 3),
                    texttemplate="%{text}"))
                fig.update_layout(height=600,
                                  title="Distance de Reynolds entre races")
                st.plotly_chart(fig, use_container_width=True,
                                key="phylo_plot_reynolds")

                newick = nj_tree_newick(st.session_state.reynolds_D,
                                        st.session_state.reynolds_pops)
                st.subheader("Arbre NJ (Newick)")
                st.code(newick, language="text")
                st.download_button("⬇ Télécharger Newick",
                                   newick.encode("utf-8"),
                                   file_name="cattle_nj_tree.nwk",
                                   mime="text/plain",
                                   key="phylo_dl_newick")
                st.info("💡 Importez dans **SplitsTree4** ou **iTOL**.")

    # ============ TAB 10 : Export ============
    with tabs[9]:
        st.subheader("📤 Export des données")
        step_header("2.3", "Export PLINK / VCF",
                    "Génération .bed/.bim/.fam ou .vcf.",
                    "Format binaire PLINK réduit la taille de 80%.",
                    "ZIP PLINK + VCF téléchargeables.")

        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            ind_use = st.session_state.ind_filt
            snp_use = (st.session_state.snp_pruned if has_pruned()
                       else st.session_state.snp_filt)

            prefix = st.text_input("Préfixe", "bovine_qc", key="export_prefix")
            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button("📦 Préparer PLINK ZIP",
                             use_container_width=True, key="export_btn_plink"):
                    try:
                        with st.spinner("Génération..."):
                            st.session_state["_plink_zip"] = build_plink_zip(
                                gt_use, ind_use, snp_use, prefix=prefix)
                        st.success("✅ Prêt.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.get("_plink_zip"):
                    st.download_button(
                        "⬇ .zip PLINK",
                        data=st.session_state["_plink_zip"],
                        file_name=f"{prefix}_plink.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="export_dl_plink")

            with c2:
                if st.button("📄 Préparer VCF", use_container_width=True,
                             key="export_btn_vcf"):
                    try:
                        with st.spinner("Génération VCF..."):
                            st.session_state["_vcf_str"] = build_vcf_output(
                                gt_use, ind_use, snp_use)
                        st.success("✅ Prêt.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.get("_vcf_str"):
                    st.download_button(
                        "⬇ .vcf",
                        data=st.session_state["_vcf_str"].encode("utf-8"),
                        file_name=f"{prefix}.vcf",
                        mime="text/plain",
                        use_container_width=True,
                        key="export_dl_vcf")

            with c3:
                st.metric("Individus", gt_use.shape[0])
                st.metric("SNPs", gt_use.shape[1])

    # ============ TAB 11 : Rapport ============
    with tabs[10]:
        st.subheader("📄 Rapport HTML")
        step_header("7", "Synthèse et rapport",
                    "Rapport HTML interactif avec figures + interprétations.",
                    "Livrable scientifique partageable.",
                    "Fichier .html téléchargeable.")

        if not has_qc():
            st.warning("⚠️ Lancez au moins le QC.")
        else:
            project_name = st.text_input("Nom du projet", "Cattle_Project",
                                         key="report_project")

            interps = {}
            if st.session_state.qc_stats:
                interps["QC"] = interpret_qc(st.session_state.qc_stats)
            if has_qc():
                interps["MAF"] = interpret_maf(maf(st.session_state.gt_filt))
            if st.session_state.fst is not None:
                fst_c = st.session_state.fst[
                    np.isfinite(st.session_state.fst)]
                if len(fst_c):
                    interps["FST"] = interpret_fst(fst_c)
            if st.session_state.cv_results:
                interps["Admixture"] = interpret_admixture(
                    st.session_state.cv_results)
            if st.session_state.froh is not None:
                interps["ROH"] = interpret_roh(st.session_state.froh)

            if st.button("📄 Générer rapport HTML",
                         use_container_width=True, key="report_btn_gen"):
                figures = {}
                if st.session_state.pca_scores is not None:
                    figures["PCA"] = plot_pca(
                        st.session_state.pca_scores,
                        st.session_state.pca_var,
                        st.session_state.ind_filt["FID"].astype(str).to_numpy())
                if st.session_state.mds_coords is not None:
                    figures["MDS"] = plot_mds(
                        st.session_state.mds_coords,
                        st.session_state.ind_filt["FID"].astype(str).to_numpy())
                if st.session_state.fst is not None:
                    figures["Manhattan FST"] = plot_manhattan(
                        st.session_state.fst,
                        st.session_state.snp_filt["CHR"].values)
                if st.session_state.fst_pairwise_matrix is not None:
                    figures["FST pairwise"] = plot_fst_pairwise(
                        st.session_state.fst_pairwise_matrix,
                        st.session_state.fst_pops)
                if st.session_state.admixture_Q is not None:
                    fig_adm, _ = plot_admixture(
                        st.session_state.admixture_Q,
                        st.session_state.ind_filt["IID"].astype(str).to_numpy(),
                        st.session_state.ind_filt["FID"].astype(str).to_numpy(),
                        st.session_state.admixture_K)
                    figures["Admixture"] = fig_adm
                if st.session_state.gwas_results is not None:
                    fig_gwas = plot_gwas_manhattan(
                        st.session_state.gwas_results,
                        title=f"GWAS — {st.session_state.gwas_pheno_name}")
                    if fig_gwas is not None:
                        figures["GWAS Manhattan"] = fig_gwas

                html = build_report_html(
                    {"project_name": project_name},
                    st.session_state.qc_stats,
                    figures=figures, interpretations=interps)
                st.session_state["_report_html"] = html
                st.success("✅ Rapport HTML prêt.")

            if st.session_state.get("_report_html"):
                st.download_button(
                    "⬇ Télécharger HTML",
                    data=st.session_state["_report_html"].encode("utf-8"),
                    file_name=f"rapport_{datetime.now():%Y%m%d_%H%M}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="report_dl_html")

            if interps:
                st.subheader("🤖 Interprétations IA incluses")
                for module, interp in interps.items():
                    with st.expander(f"{module} — {interp['verdict']}"):
                        st.markdown(f"**{interp['explanation']}**")
                        st.json(interp["metrics"])
                        st.info(interp["recommendation"])

    # ---------- PIPELINE COMPLET ----------
    if st.session_state.get("run_requested"):
        st.session_state.run_requested = False
        try:
            with st.spinner("🚀 Pipeline complet..."):
                gt_a, snp_a = filter_autosomes(st.session_state.gt,
                                               st.session_state.snp_df)
                st.session_state.gt = gt_a
                st.session_state.snp_df = snp_a

                params = {"geno": geno, "mind": mind, "maf": maf_thr,
                          "hwe": hwe_thr, "het_sd": het_sd}
                gt_f, ind_f, snp_f, qc_stats = apply_qc_filters(
                    gt_a, st.session_state.ind_df, snp_a, params)
                st.session_state.gt_filt = gt_f
                st.session_state.ind_filt = ind_f
                st.session_state.snp_filt = snp_f
                st.session_state.qc_stats = qc_stats

                keep = ld_pruning(gt_f, window=50, step=5, r2_thr=ld_r2)
                st.session_state.gt_pruned = gt_f[:, keep]
                st.session_state.snp_pruned = snp_f[keep].reset_index(drop=True)
                gt_p = st.session_state.gt_pruned
                snp_p = st.session_state.snp_pruned

                fid_arr = ind_f["FID"].astype(str).to_numpy()

                st.session_state.pca_scores, st.session_state.pca_var = \
                    pca_analysis(gt_p, 10)
                st.session_state.mds_coords = mds_analysis(gt_p, 5)
                st.session_state.king_matrix = king_kinship(gt_p)

                st.session_state.ld_df = ld_decay(gt_p, snp_p["BP"].values,
                                                  max_kb=1000, max_snp=1000)

                st.session_state.fst = fst_per_snp(gt_p, fid_arr)
                mat, pops = fst_pairwise(gt_p, fid_arr)
                st.session_state.fst_pairwise_matrix = mat
                st.session_state.fst_pops = pops

                cv = admix_cv_error(gt_p, K_range=(2, 6), n_reps=2)
                st.session_state.cv_results = cv
                ks = sorted(cv.keys())
                K_opt = ks[int(np.argmin([cv[k]["mean"] for k in ks]))]
                Q, _ = admixture_nmf(gt_p, K=K_opt)
                st.session_state.admixture_Q = Q
                st.session_state.admixture_K = K_opt

                snp_json = snp_p[["CHR", "BP"]].to_json()
                roh_df, froh = detect_roh(gt_p, snp_json)
                st.session_state.roh_df = roh_df
                st.session_state.froh = froh

                ne_dict = {}
                for fid in ind_f["FID"].unique():
                    mask = (ind_f["FID"] == fid).values
                    if mask.sum() < 5:
                        continue
                    ne_dict[str(fid)] = estimate_ne_historical(
                        gt_p[mask], snp_p)
                st.session_state.ne_dict = ne_dict

                D, rp = reynolds_distance(gt_p, fid_arr)
                st.session_state.reynolds_D = D
                st.session_state.reynolds_pops = rp

            st.success("✅ Pipeline complet terminé ! Consultez les onglets.")
        except Exception as e:
            st.error(f"❌ Erreur pipeline : {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
