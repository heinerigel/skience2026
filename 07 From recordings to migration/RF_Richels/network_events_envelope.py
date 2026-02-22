import os
import glob
import pandas as pd
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy import signal
from collections import Counter
import warnings
import sys
import time
from datetime import datetime
from multiprocessing import Pool
import gc

warnings.filterwarnings("ignore")

# ================= CONFIG =================
CLASSIFIED_DIR = "./single_stations/classified"
OUTPUT_DIR = "./single_stations/network_events"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WAVEFORM_WINDOW = 10.0
TIME_TOLERANCE = 7.0
XCORR_THRESHOLD = 0.7
MIN_STATIONS = 4
REQUIRE_SAME_LABEL = False

NETWORK = "4D"
CHANNEL = "??Z"
LOCATION = "*"
FDSN_URL = "http://tarzan.geophysik.uni-muenchen.de"

LOG_EVERY = 500
NUM_WORKERS = 10 # Careful when hardcoded!

# MEMORY CONTROL
DTYPE = np.float32
CACHE_SIZE_LIMIT = 1000  # Max waveforms to cache per worker
BATCH_SIZE = 100  # Process and write results in batches

# ============== HELPERS ===============
def log(msg):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def load_classified_triggers(csv_file):
    """Load only necessary columns to reduce memory"""
    df = pd.read_csv(
        csv_file,
        usecols=["trigger_time", "predicted_event"],
        dtype={"predicted_event": "category"}  # Use categorical for labels
    )
    df["trigger_time"] = pd.to_datetime(
        df["trigger_time"], format="mixed", utc=True
    )
    return df.sort_values("trigger_time").reset_index(drop=True)

def fetch_waveform(station, t):
    client = Client(FDSN_URL)
    try:
        st = client.get_waveforms(
            network=NETWORK,
            station=station,
            location=LOCATION,
            channel=CHANNEL,
            starttime=t - WAVEFORM_WINDOW / 2,
            endtime=t + WAVEFORM_WINDOW / 2,
        )
        if not st:
            return None

        st.detrend("linear")
        st.taper(0.05)
        st.filter("bandpass", freqmin=1, freqmax=20)

        tr = st.select(component="Z")
        if not tr:
            return None

        # Convert to float32
        data = tr[0].data.astype(DTYPE, copy=False)

        # Compute envelope in-place where possible
        analytic_signal = signal.hilbert(data)
        envelope = np.abs(analytic_signal).astype(DTYPE, copy=False)

        # Normalize envelope
        mean_val = envelope.mean()
        std_val = envelope.std()
        envelope -= mean_val
        envelope /= (std_val + 1e-9)

        # Explicit cleanup
        del st, tr, analytic_signal, data
        
        return envelope

    except Exception:
        return None

def normalized_xcorr(env1, env2):
    """Memory-efficient cross-correlation"""
    n = min(len(env1), len(env2))
    if n == 0:
        return 0.0

    # Use views instead of copies
    e1 = env1[:n]
    e2 = env2[:n]
    
    # Use 'auto' method to let scipy choose most efficient
    corr = signal.correlate(e1, e2, mode="same", method="auto")
    val = float(np.max(np.abs(corr)) / n)
    
    del corr
    return val

class LRUCache:
    """Simple LRU cache with size limit"""
    def __init__(self, max_size):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)

def process_ref_trigger(args):
    i, ref_row, ref_station, ref_df, station_dfs = args

    local_cache = LRUCache(CACHE_SIZE_LIMIT)

    def get_cached_waveform_local(key, station, time):
        cached = local_cache.get(key)
        if cached is not None:
            return cached
        wf = fetch_waveform(station, time)
        if wf is not None:
            local_cache.put(key, wf)
        return wf

    t_ref = UTCDateTime(ref_row["trigger_time"])
    key_ref = f"{ref_station}_{t_ref.timestamp}"
    w_ref = get_cached_waveform_local(key_ref, ref_station, t_ref)
    if w_ref is None:
        return None

    stations = [ref_station]
    labels = [ref_row["predicted_event"]]

    xcorr_attempts = 0
    xcorr_success = 0

    for st, df in station_dfs.items():
        if st == ref_station:
            continue
        
        # Use numpy for faster comparison
        dt = np.abs((df["trigger_time"] - ref_row["trigger_time"]).dt.total_seconds())
        mask = dt <= TIME_TOLERANCE

        if REQUIRE_SAME_LABEL:
            mask &= (df["predicted_event"] == ref_row["predicted_event"]).values

        candidates = df[mask]
        station_matched = False
        
        for _, row in candidates.iterrows():
            t_other = UTCDateTime(row["trigger_time"])
            key_other = f"{st}_{t_other.timestamp}"
            w_other = get_cached_waveform_local(key_other, st, t_other)
            if w_other is None:
                continue
            xcorr_attempts += 1
            if normalized_xcorr(w_ref, w_other) >= XCORR_THRESHOLD:
                xcorr_success += 1
                station_matched = True
                labels.append(row["predicted_event"])
                break
        
        if station_matched:
            stations.append(st)

    # Cleanup
    del w_ref
    
    if len(stations) >= MIN_STATIONS:
        consensus = Counter(labels).most_common(1)[0][0]
        return {
            "event_time": str(t_ref),
            "label": consensus,
            "n_stations": len(stations),
            "stations": ",".join(stations),
            "xcorr_attempts": xcorr_attempts,
            "xcorr_success": xcorr_success
        }
    return None

# =================== MAIN ==================
log("NETWORK EVENT DETECTION STARTED")

files = sorted(glob.glob(os.path.join(CLASSIFIED_DIR, "*_classified.csv")))
if not files:
    sys.exit("No classified CSV files found")

station_dfs = {}
for f in files:
    station = os.path.basename(f).split("_")[0].split(".")[-1]
    df = load_classified_triggers(f)
    if len(df):
        station_dfs[station] = df
        log(f"{station}: {len(df)} triggers")

ref_station = max(station_dfs, key=lambda s: len(station_dfs[s]))
ref_df = station_dfs[ref_station]
log(f"Reference station: {ref_station} ({len(ref_df)})")

# Prepare arguments
args_list = [(i, row, ref_station, ref_df, station_dfs) for i, row in ref_df.iterrows()]

network_events = []
total_xcorr_attempts = 0
total_xcorr_success = 0
start_time = time.time()

log(f"Using {NUM_WORKERS} CPU cores")

# Open output file for incremental writing
out_file = os.path.join(OUTPUT_DIR, "network_envelope_0.7_summary.csv")
with open(out_file, 'w') as f:
    f.write("event_time,label,n_stations,stations\n")

batch_results = []

with Pool(processes=NUM_WORKERS) as pool:
    for idx, result in enumerate(pool.imap_unordered(process_ref_trigger, args_list, chunksize=5)):
        if result:
            batch_results.append({
                "event_time": result["event_time"],
                "label": result["label"],
                "n_stations": result["n_stations"],
                "stations": result["stations"]
            })
            total_xcorr_attempts += result["xcorr_attempts"]
            total_xcorr_success += result["xcorr_success"]
            network_events.append(result)

        # Write batch to disk and clear memory
        if len(batch_results) >= BATCH_SIZE:
            batch_df = pd.DataFrame(batch_results)
            batch_df.to_csv(out_file, mode='a', header=False, index=False)
            batch_results.clear()
            gc.collect()

        if idx > 0 and idx % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(ref_df) - idx) / rate / 3600 if rate > 0 else np.nan
            success = 100 * total_xcorr_success / max(1, total_xcorr_attempts)
            log(
                f"{idx+1}/{len(ref_df)} ({100*(idx+1)/len(ref_df):.2f}%) | "
                f"Events: {len(network_events)} | "
                f"Rate: {rate:.2f} trig/s | "
                f"ETA: {remaining:.2f} h | "
                f"XCorr success: {success:.1f}%"
            )

# Write remaining batch
if batch_results:
    batch_df = pd.DataFrame(batch_results)
    batch_df.to_csv(out_file, mode='a', header=False, index=False)
    batch_results.clear()

log("PROCESSING COMPLETE")
log(f"Network events: {len(network_events)}")
log(f"XCorr attempts: {total_xcorr_attempts}")
log(f"XCorr success:  {total_xcorr_success}")
log(f"Runtime: {(time.time() - start_time)/3600:.2f} h")
log(f"Saved → {out_file}")