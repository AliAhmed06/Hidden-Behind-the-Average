# =========================================================
# Cross-Resource Calibration Audit for Fetal Ultrasound Plane Classification
# Spain (FETAL_PLANES_DB) -> Africa (five-country low-resource dataset)
# Pooled vs. Class-Conditional (Mondrian) Split Conformal Prediction
# LOCAL-UPLOAD VERSION — paths confirmed against your dataset structure
# =========================================================

import os, math, random, json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIG — confirmed paths
# ---------------------------------------------------------
SPAIN_ROOT  = "/kaggle/input/datasets/aliiahmadd/fetal-planes-zenodo"
AFRICA_ROOT = "/kaggle/input/datasets/aliiahmadd/zenodo-dataset"

WORK_DIR = "/kaggle/working/fetal_audit"
os.makedirs(WORK_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

IMG_SIZE = 224
BATCH_SIZE = 32
N_EPOCHS = 12
N_SEEDS = 3
N_CALIB_REPEATS = 50
ALPHA = 0.10
MIN_CELL_N = 10
CLASSES = ["Abdomen", "Brain", "Femur", "Thorax"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

random.seed(0); np.random.seed(0); torch.manual_seed(0)

# ---------------------------------------------------------
# STEP 1: LOCATE FILES
# ---------------------------------------------------------
def build_file_index(root_dir, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                key = os.path.splitext(fn)[0].lower()
                index[key] = os.path.join(dirpath, fn)
    return index

def find_meta_files(root_dir):
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".csv") or fn.lower().endswith(".xlsx"):
                out.append(os.path.join(dirpath, fn))
    return out

assert os.path.isdir(SPAIN_ROOT), f"SPAIN_ROOT not found: {SPAIN_ROOT}"
assert os.path.isdir(AFRICA_ROOT), f"AFRICA_ROOT not found: {AFRICA_ROOT}"

spain_index = build_file_index(SPAIN_ROOT)
africa_index = build_file_index(AFRICA_ROOT)
print(f"Spain images found: {len(spain_index)} | Africa images found: {len(africa_index)}")

spain_csvs = [p for p in find_meta_files(SPAIN_ROOT) if p.lower().endswith(".csv")]
africa_csvs = [p for p in find_meta_files(AFRICA_ROOT) if p.lower().endswith(".csv")]
print("Spain metadata file:", spain_csvs[0])
print("Africa metadata file:", africa_csvs[0])

spain_df_raw = pd.read_csv(spain_csvs[0], sep=None, engine="python")
africa_df_raw = pd.read_csv(africa_csvs[0], sep=None, engine="python")

# ---------------------------------------------------------
# STEP 2: COLUMN MAPPING — confirmed against your printed CSV structure
# ---------------------------------------------------------
SPAIN_IMG_COL     = "Image_name"
SPAIN_LABEL_COL   = "Plane"
SPAIN_SPLIT_COL   = "Train "     # note: trailing space, confirmed from your printout
SPAIN_PATIENT_COL = "Patient_num"

AFRICA_IMG_COL     = "Filename"
AFRICA_LABEL_COL   = "Plane"
AFRICA_COUNTRY_COL = "Center"
AFRICA_PATIENT_COL = "Patient_num"

def normalize_label(raw):
    s = str(raw).lower()
    if "abdom" in s: return "Abdomen"
    if "brain" in s: return "Brain"
    if "femur" in s: return "Femur"
    if "thorax" in s: return "Thorax"
    return None  # Other / Maternal cervix -> dropped

# ---------------------------------------------------------
# STEP 3: BUILD UNIFIED, HARMONIZED DATAFRAMES
# ---------------------------------------------------------
def resolve_path(index, name):
    key = os.path.splitext(str(name))[0].lower()
    return index.get(key, None)

spain_df = spain_df_raw.copy()
spain_df["label"] = spain_df[SPAIN_LABEL_COL].apply(normalize_label)
spain_df = spain_df.dropna(subset=["label"]).reset_index(drop=True)
spain_df["filepath"] = spain_df[SPAIN_IMG_COL].apply(lambda n: resolve_path(spain_index, n))
spain_df = spain_df.dropna(subset=["filepath"]).reset_index(drop=True)
spain_df["is_train"] = spain_df[SPAIN_SPLIT_COL].astype(str).str.strip().isin(["1", "1.0", "True", "train"])

print("\nSpain harmonized class counts:\n", spain_df["label"].value_counts())
print("Spain train/test counts:\n", spain_df["is_train"].value_counts())
assert len(spain_df) > 0, "No Spain rows matched to image files"

africa_df = africa_df_raw.copy()
africa_df["label"] = africa_df[AFRICA_LABEL_COL].apply(normalize_label)
africa_df = africa_df.dropna(subset=["label"]).reset_index(drop=True)
africa_df["filepath"] = africa_df[AFRICA_IMG_COL].apply(lambda n: resolve_path(africa_index, n))
africa_df = africa_df.dropna(subset=["filepath"]).reset_index(drop=True)
africa_df["country"] = africa_df[AFRICA_COUNTRY_COL].astype(str).str.capitalize()

print("\nAfrica harmonized class counts:\n", africa_df["label"].value_counts())
print("Africa country x class:\n", pd.crosstab(africa_df["country"], africa_df["label"]))
assert len(africa_df) > 0, "No Africa rows matched to image files"

# ---------------------------------------------------------
# STEP 4: DATASET / DATALOADER
# ---------------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class PlaneDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["filepath"].tolist()
        self.labels = [CLASS_TO_IDX[l] for l in df["label"].tolist()]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

spain_train_df = spain_df[spain_df["is_train"]].reset_index(drop=True)
spain_test_df  = spain_df[~spain_df["is_train"]].reset_index(drop=True)
spain_fit_df, spain_val_df = train_test_split(
    spain_train_df, test_size=0.15, random_state=0, stratify=spain_train_df["label"]
)

print(f"\nSpain fit={len(spain_fit_df)} val={len(spain_val_df)} in-domain test={len(spain_test_df)}")
print(f"Africa total (calibration/test only, never used for training)={len(africa_df)}")

class_counts = spain_fit_df["label"].value_counts()
sample_weights = spain_fit_df["label"].map(lambda l: 1.0 / class_counts[l]).values
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

fit_loader = DataLoader(PlaneDataset(spain_fit_df, train_tf), batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(PlaneDataset(spain_val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
spain_test_loader = DataLoader(PlaneDataset(spain_test_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
africa_loader_all = DataLoader(PlaneDataset(africa_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------------------------------------------------------
# STEP 5: MODEL
# ---------------------------------------------------------
def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feat = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_feat, len(CLASSES))
    return m.to(DEVICE)

def train_one_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    crit = nn.CrossEntropyLoss()
    best_val_acc, best_state = 0.0, None

    for epoch in range(N_EPOCHS):
        model.train()
        running_loss = 0.0
        for x, y in fit_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        sched.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item(); total += y.size(0)
        val_acc = correct / total
        print(f"  seed {seed} epoch {epoch+1}/{N_EPOCHS} loss={running_loss/len(spain_fit_df):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(WORK_DIR, f"model_seed{seed}.pt"))
    return model, best_val_acc

# ---------------------------------------------------------
# STEP 6: TEMPERATURE SCALING + ECE
# ---------------------------------------------------------
def fit_temperature(logits, labels):
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / T.clamp(min=0.05), labels_t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.clamp(min=0.05).item())

def expected_calibration_error(probs, labels, n_bins=10):
    confidences = probs.max(1)
    predictions = probs.argmax(1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return ece

# ---------------------------------------------------------
# STEP 7: CONFORMAL PREDICTION (LAC split-conformal + Mondrian by class)
# ---------------------------------------------------------
def lac_scores(probs, labels):
    return 1.0 - probs[np.arange(len(labels)), labels]

def conformal_quantile(scores, alpha):
    n = len(scores)
    if n == 0:
        return np.nan
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, q_level, method="higher")

def prediction_sets(probs, qhat):
    return probs >= (1.0 - qhat)

def coverage_and_size(pred_sets, labels):
    covered = pred_sets[np.arange(len(labels)), labels]
    return covered.mean(), pred_sets.sum(1).mean(), covered

def bootstrap_ci(values, n_boot=2000, ci=0.90):
    values = np.asarray(values)
    if len(values) == 0:
        return (np.nan, np.nan)
    boots = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi

# ---------------------------------------------------------
# STEP 8: MAIN EXPERIMENT LOOP (multi-seed)
# ---------------------------------------------------------
all_seed_results = []

for seed in range(N_SEEDS):
    print(f"\n===== SEED {seed} =====")
    model, val_acc = train_one_seed(seed)

    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            val_logits.append(model(x.to(DEVICE)).cpu().numpy())
            val_labels.append(y.numpy())
    val_logits = np.concatenate(val_logits); val_labels = np.concatenate(val_labels)
    T = fit_temperature(val_logits, val_labels)
    print(f"  fitted temperature T={T:.3f}")

    def get_probs_labels(loader):
        probs_list, labels_list = [], []
        with torch.no_grad():
            for x, y in loader:
                logits = model(x.to(DEVICE)).cpu().numpy()
                probs = torch.softmax(torch.tensor(logits) / T, dim=1).numpy()
                probs_list.append(probs); labels_list.append(y.numpy())
        return np.concatenate(probs_list), np.concatenate(labels_list)

    spain_test_probs, spain_test_labels = get_probs_labels(spain_test_loader)
    africa_probs, africa_labels = get_probs_labels(africa_loader_all)

    spain_acc = (spain_test_probs.argmax(1) == spain_test_labels).mean()
    africa_acc = (africa_probs.argmax(1) == africa_labels).mean()
    spain_ece = expected_calibration_error(spain_test_probs, spain_test_labels)
    africa_ece = expected_calibration_error(africa_probs, africa_labels)
    print(f"  Spain in-domain acc={spain_acc:.4f} ECE={spain_ece:.4f}")
    print(f"  Africa OOD acc={africa_acc:.4f} ECE={africa_ece:.4f} (gap={spain_acc-africa_acc:.4f})")

    africa_df_reset = africa_df.reset_index(drop=True)
    for rep in range(N_CALIB_REPEATS):
        idx = np.arange(len(africa_df_reset))
        try:
            strat_key = africa_df_reset.loc[idx, "label"] + "_" + africa_df_reset.loc[idx, "country"]
            calib_idx, test_idx = train_test_split(idx, test_size=0.5, random_state=rep, stratify=strat_key)
        except ValueError:
            calib_idx, test_idx = train_test_split(idx, test_size=0.5, random_state=rep)

        calib_probs, calib_labels = africa_probs[calib_idx], africa_labels[calib_idx]
        test_probs, test_labels = africa_probs[test_idx], africa_labels[test_idx]
        test_meta = africa_df_reset.iloc[test_idx].reset_index(drop=True)

        qhat_pooled = conformal_quantile(lac_scores(calib_probs, calib_labels), ALPHA)
        sets_pooled = prediction_sets(test_probs, qhat_pooled)
        cov_pooled_overall, size_pooled_overall, covered_pooled = coverage_and_size(sets_pooled, test_labels)

        qhat_by_class = {}
        for c_idx in range(len(CLASSES)):
            mask = calib_labels == c_idx
            if mask.sum() >= 2:
                qhat_by_class[c_idx] = conformal_quantile(lac_scores(calib_probs[mask], calib_labels[mask]), ALPHA)
        sets_classcond = np.zeros_like(sets_pooled)
        for i in range(len(test_labels)):
            q = qhat_by_class.get(test_labels[i], qhat_pooled)
            sets_classcond[i] = test_probs[i] >= (1 - q)
        covered_classcond = sets_classcond[np.arange(len(test_labels)), test_labels]

        for country in africa_df_reset["country"].unique():
            for c_idx, c_name in enumerate(CLASSES):
                cell_mask = (test_meta["country"].values == country) & (test_labels == c_idx)
                n_cell = int(cell_mask.sum())
                if n_cell < 1:
                    continue
                all_seed_results.append({
                    "seed": seed, "repeat": rep, "domain": "Africa",
                    "country": country, "class": c_name, "n_test_cell": n_cell,
                    "coverage_pooled_threshold": covered_pooled[cell_mask].mean(),
                    "coverage_classcond_threshold": covered_classcond[cell_mask].mean(),
                    "avg_set_size_classcond": sets_classcond[cell_mask].sum(1).mean(),
                })

        all_seed_results.append({
            "seed": seed, "repeat": rep, "domain": "Africa_OVERALL",
            "country": "ALL", "class": "ALL", "n_test_cell": int(len(test_labels)),
            "coverage_pooled_threshold": cov_pooled_overall,
            "coverage_classcond_threshold": covered_classcond.mean(),
            "avg_set_size_classcond": sets_classcond.sum(1).mean(),
        })

    all_seed_results.append({
        "seed": seed, "repeat": -1, "domain": "Spain_test", "country": "Spain", "class": "ALL",
        "n_test_cell": int(len(spain_test_labels)),
        "coverage_pooled_threshold": np.nan, "coverage_classcond_threshold": np.nan,
        "avg_set_size_classcond": np.nan,
        "spain_accuracy": spain_acc, "africa_accuracy": africa_acc,
        "spain_ece": spain_ece, "africa_ece": africa_ece, "temperature": T,
    })

results_df = pd.DataFrame(all_seed_results)
results_df.to_csv(os.path.join(WORK_DIR, "raw_conformal_results.csv"), index=False)

# ---------------------------------------------------------
# STEP 9: AGGREGATE + FLAG FAILURES
# ---------------------------------------------------------
cell_results = results_df[results_df["domain"] == "Africa"].copy()
summary_rows = []
for (country, cls), g in cell_results.groupby(["country", "class"]):
    n_med = g["n_test_cell"].median()
    flag = "OK" if n_med >= MIN_CELL_N else "INSUFFICIENT_DATA"
    lo_p, hi_p = bootstrap_ci(g["coverage_pooled_threshold"].dropna().values)
    lo_c, hi_c = bootstrap_ci(g["coverage_classcond_threshold"].dropna().values)
    summary_rows.append({
        "country": country, "class": cls, "median_n_test": n_med,
        "mean_coverage_pooled": g["coverage_pooled_threshold"].mean(),
        "coverage_pooled_90CI_lo": lo_p, "coverage_pooled_90CI_hi": hi_p,
        "mean_coverage_classcond": g["coverage_classcond_threshold"].mean(),
        "coverage_classcond_90CI_lo": lo_c, "coverage_classcond_90CI_hi": hi_c,
        "mean_set_size_classcond": g["avg_set_size_classcond"].mean(),
        "target_coverage": 1 - ALPHA,
        "undercoverage_gap_pooled": (1 - ALPHA) - g["coverage_pooled_threshold"].mean(),
        "data_flag": flag,
    })
summary_df = pd.DataFrame(summary_rows).sort_values("undercoverage_gap_pooled", ascending=False)
summary_df.to_csv(os.path.join(WORK_DIR, "coverage_summary_by_group.csv"), index=False)

print("\n================ WORST-COVERED (country, class) GROUPS ================")
print(summary_df.to_string(index=False))

overall = results_df[results_df["domain"] == "Africa_OVERALL"]
print(f"\nOverall Africa coverage, pooled threshold: {overall['coverage_pooled_threshold'].mean():.3f} (target {1-ALPHA:.2f})")
print(f"Overall Africa coverage, class-conditional threshold: {overall['coverage_classcond_threshold'].mean():.3f} (target {1-ALPHA:.2f})")

acc_rows = results_df[results_df["domain"] == "Spain_test"]
print("\nPer-seed accuracy / ECE (Spain in-domain vs Africa OOD):")
print(acc_rows[["seed","spain_accuracy","africa_accuracy","spain_ece","africa_ece","temperature"]].to_string(index=False))

# ---------------------------------------------------------
# STEP 10: PLOT
# ---------------------------------------------------------
plot_df = summary_df[summary_df["data_flag"] == "OK"]
if len(plot_df) > 0:
    plt.figure(figsize=(10, 5))
    x_labels = plot_df["country"] + " / " + plot_df["class"]
    plt.bar(x_labels, plot_df["mean_coverage_pooled"], label="Pooled threshold")
    plt.bar(x_labels, plot_df["mean_coverage_classcond"], alpha=0.5, label="Class-conditional threshold")
    plt.axhline(1 - ALPHA, color="red", linestyle="--", label=f"Target coverage ({1-ALPHA:.0%})")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Empirical coverage")
    plt.title("Conformal coverage by (country, class): pooled vs class-conditional calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, "coverage_by_group.png"), dpi=150)
    plt.show()

print(f"\nAll outputs saved in: {WORK_DIR}")