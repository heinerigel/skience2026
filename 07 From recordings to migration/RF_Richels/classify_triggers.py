"""
Script 1: classify_triggers.py
Applies Random Forest classification to station trigger features
Run this once to classify all triggers, then use detect_network_events.py or network_events_envelope.py
"""

import os
import glob
import pandas as pd
import joblib
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ================= CONFIG =================
TRIGGERS_DIR = "./single_stations"
OUTPUT_DIR = "./single_stations/classified"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_LABELS = {"icequake", "surface_waves", "short_signal"}

RF_MODEL_PATH = "random_forest_model_single_stations.pkl"
LE_PATH = "label_encoder.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"

# ================= LOAD MODEL =================
print("Loading Random Forest model...")
clf = joblib.load(RF_MODEL_PATH)
le = joblib.load(LE_PATH)
feature_cols = joblib.load(FEATURE_NAMES_PATH)

print(f"Loaded RF model with {len(feature_cols)} features")
print(f"Allowed labels: {ALLOWED_LABELS}")

# ================= FUNCTIONS =================
def load_features(csv_file):
    """Load features from CSV and ensure trigger_time column exists"""
    df = pd.read_csv(csv_file, sep=",")
    for col in ["event_time", "trigger_time", "time", "utc_time"]:
        if col in df.columns:
            df["trigger_time"] = pd.to_datetime(df[col], utc=True)
            return df
    raise ValueError(f"No time column found in {csv_file}")

def classify_triggers(df):
    """Apply Random Forest classifier and filter by allowed labels"""
    X = df.reindex(columns=feature_cols, fill_value=0)
    preds = clf.predict(X)
    df["predicted_event"] = le.inverse_transform(preds)
    df = df[df["predicted_event"].isin(ALLOWED_LABELS)].reset_index(drop=True)
    return df

def save_station_triggers(df, station):
    """Save classified triggers for a station"""
    out_csv = os.path.join(OUTPUT_DIR, f"{station}_classified.csv")
    df.to_csv(out_csv, index=False)
    print(f"  Saved → {out_csv}")

# ================= MAIN =================
all_csvs = sorted(glob.glob(os.path.join(TRIGGERS_DIR, "*_features.csv")))
print(f"\nFound {len(all_csvs)} station feature files")

print("\n" + "="*60)
print("CLASSIFYING TRIGGERS")
print("="*60)

total_triggers = 0
total_classified = 0

for csv_file in all_csvs:
    station = os.path.basename(csv_file).split("_")[0]
    print(f"\nStation {station}:")

    df = load_features(csv_file)
    n_original = len(df)
    
    df = classify_triggers(df)
    n_classified = len(df)
    
    print(f"  Original triggers: {n_original}")
    print(f"  After filtering: {n_classified}")
    
    if n_classified > 0:
        print(f"  Labels: {dict(Counter(df['predicted_event']))}")
        save_station_triggers(df, station)
        total_classified += n_classified
    else:
        print(f"  No valid triggers for this station")
    
    total_triggers += n_original

print("\n" + "="*60)
print("CLASSIFICATION COMPLETE")
print("="*60)
print(f"Total triggers processed: {total_triggers}")
print(f"Total triggers classified: {total_classified}")
print(f"Output directory: {OUTPUT_DIR}")
print("\nNext step: Run detect_network_events.py or network_events_envelope.py to find network events")