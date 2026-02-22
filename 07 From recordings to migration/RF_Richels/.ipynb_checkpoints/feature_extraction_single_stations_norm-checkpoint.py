#!/usr/bin/env python3
import os
import time
import logging
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy import signal
from scipy.signal import hilbert, find_peaks
from scipy.stats import skew, kurtosis
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# ================= CONFIG =================
input_file  = "./single_stations/4D.RC03_just_triggers.csv"
output_file = "./single_stations/4D.RC03_features.csv"

window_before = 1.0
window_after  = 5.0
buffer_time   = 5.0 

station = "RC03"
network = "4D"
channel = "*Z"
FDSN_URL = "http://tarzan.geophysik.uni-muenchen.de"

MAX_WORKERS = min(8, os.cpu_count() or 1)
LOG_EVERY = 10

# ============ LOGGING ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger()

# =============== HELPER FUNCTIONS =================
def format_time(seconds):
    """Format seconds into readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def calculate_spectrum(data, sr, n):
    """Returns positive frequencies and normalized PSD."""
    win = data * signal.windows.hann(n)
    fft = np.abs(np.fft.rfft(win))[1:]
    freqs = np.fft.rfftfreq(n, 1 / sr)[1:]
    
    if np.sum(fft) > 0:
        P = fft / np.sum(fft)
    else:
        P = np.zeros_like(fft)
    
    return freqs, fft, P

def spectral_rolloff(freqs, fft, percentile=0.85):
    """Frequency below which X% of spectral energy is contained (normalized to 20 Hz)."""
    cumsum = np.cumsum(fft**2)
    total = cumsum[-1]
    if total > 0:
        threshold = percentile * total
        idx = np.where(cumsum >= threshold)[0]
        if len(idx) > 0:
            return freqs[idx[0]] / 20.0
    return 0

def spectral_flatness(fft):
    """Measure of noisiness vs tonality. Close to 1 = noise-like."""
    if len(fft) == 0 or np.any(fft <= 0):
        return 0
    geometric_mean = np.exp(np.mean(np.log(fft + 1e-12)))
    arithmetic_mean = np.mean(fft)
    return geometric_mean / (arithmetic_mean + 1e-12)

def calculate_spectrogram_features(data, sr):
    """Extract features from time-frequency representation."""
    nperseg = min(256, len(data) // 4)
    if nperseg < 16:
        return {}
    
    f, t, Sxx = signal.spectrogram(data, sr, nperseg=nperseg, noverlap=nperseg//2)
    
    # Frequency of maximum energy over time
    freq_track = f[np.argmax(Sxx, axis=0)]
    
    features = {}
    # Coefficient of variation (std/mean)
    mean_freq = np.mean(freq_track)
    features["freq_track_cv"] = np.std(freq_track) / (mean_freq + 1e-12)  # Frequency modulation
    features["freq_track_range_norm"] = (np.max(freq_track) - np.min(freq_track)) / (mean_freq + 1e-12)  # Freq sweep range
    
    # Spectrogram concentration
    Sxx_norm = Sxx / (np.sum(Sxx) + 1e-12)
    features["tf_concentration"] = np.sum(Sxx_norm ** 2)  # Time-frequency focus
    
    return features

def calculate_recurrence_features(data, sr):
    """Phase space recurrence features for signal regularity."""
    # Downsample for efficiency or it takes really long
    if len(data) > 500:
        step = len(data) // 250
        data_ds = data[::step]
    else:
        data_ds = data
    
    if len(data_ds) < 10:
        return {}
    
    # Normalize to zero mean, unit variance
    data_norm = (data_ds - np.mean(data_ds)) / (np.std(data_ds) + 1e-12)
    
    # Time-delay embedding
    tau = max(1, int(0.01 * sr))  # Approximate delay
    m = 3  # Embedding dimension
    
    n_points = len(data_norm) - (m-1) * tau
    if n_points < 5:
        return {}
    
    # Create embedded vectors
    embedded = np.zeros((n_points, m))
    for i in range(m):
        embedded[:, i] = data_norm[i*tau:i*tau + n_points]
    
    # Calculate recurrence distances
    dists = np.zeros((min(50, n_points), min(50, n_points)))
    n_calc = min(50, n_points)
    
    for i in range(n_calc):
        for j in range(i, n_calc):
            dists[i, j] = np.linalg.norm(embedded[i] - embedded[j])
            dists[j, i] = dists[i, j]
    
    features = {}
    # Normalize distances by embedding dimension
    features["mean_recurrence_dist_norm"] = np.mean(dists) / np.sqrt(m)  # Phase space complexity
    
    return features

def calculate_wavelet_features(data, sr):
    """Wavelet-based features for multi-scale analysis, lightweight."""
    try:
        # Use a much faster approximation based on bandpass filtering
        # This mimics wavelet decomposition but is 10-100x faster
        
        # Define frequency bands that approximate wavelet scales
        # Each band roughly corresponds to a wavelet scale
	# Needed for efficiency
        bands = [
            (1, 3),    # Low frequency / large scale
            (3, 6),
            (6, 10),
            (10, 15),
            (15, 20)   # High frequency / small scale
        ]
        
        features = {}
        band_energies = []
        
        # Calculate energy in each band (approximates wavelet scale energy)
        for low, high in bands:
            # Simple bandpass using FFT (very fast)
            fft = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1/sr)
            
            # Zero out frequencies outside band
            fft_filtered = fft.copy()
            fft_filtered[(freqs < low) | (freqs > high)] = 0
            
            # Energy in this band
            energy = np.sum(np.abs(fft_filtered)**2)
            band_energies.append(energy)
        
        band_energies = np.array(band_energies)
        total_energy = np.sum(band_energies) + 1e-12
        
        # Dominant scale (which band has most energy)
        features["wavelet_peak_scale"] = np.argmax(band_energies) / len(bands)
        
        # Low vs high frequency energy ratio
        low_energy = np.sum(band_energies[:2])  # First 2 bands
        high_energy = np.sum(band_energies[-2:])  # Last 2 bands
        features["wavelet_scale_ratio"] = low_energy / (high_energy + 1e-12)
        
        # Energy concentration (how focused is energy in one band vs spread out)
        features["wavelet_time_concentration"] = np.max(band_energies) / total_energy
        
        return features
    except Exception as e:
        logger.warning(f"Wavelet features failed: {e}")
        return {}

def calculate_autocorr_features(data, sr):
    """Autocorrelation-based periodicity features."""
    if len(data) < 20:
        return {}
    
    # Normalize to zero mean, unit variance
    data_norm = (data - np.mean(data)) / (np.std(data) + 1e-12)
    
    # Autocorrelation
    max_lag = min(len(data) // 2, int(2 * sr))  # Up to 2 seconds
    acf = np.correlate(data_norm, data_norm, mode='full')
    acf = acf[len(acf)//2:][:max_lag]
    acf = acf / (acf[0] + 1e-12)  # Normalize
    
    features = {}
    
    # First zero crossing, normalized by max lag
    zero_crossings = np.where(np.diff(np.sign(acf)))[0]
    if len(zero_crossings) > 0:
        first_zero = zero_crossings[0]
        features["acf_first_zero_norm"] = first_zero / max_lag  # Dominant period
    else:
        features["acf_first_zero_norm"] = 1.0
    
    # Decay rate, normalized by max lag
    if len(acf) > 10:
        # Find where ACF drops to 1/e
        threshold_idx = np.where(acf < 1/np.e)[0]
        if len(threshold_idx) > 0:
            features["acf_decay_norm"] = threshold_idx[0] / max_lag  # Signal persistence
        else:
            features["acf_decay_norm"] = 1.0
    else:
        features["acf_decay_norm"] = 1.0
    
    return features

# ================= FEATURE EXTRACTION =================
def extract_features(trace):
    data = trace.data.astype(np.float64)
    data -= np.mean(data)
    sr = trace.stats.sampling_rate
    n = len(data)
    features = {}

    # Normalization references
    rms = np.sqrt(np.mean(data**2)) + 1e-12
    mean_abs = np.mean(np.abs(data)) + 1e-12
    max_abs = np.max(np.abs(data)) + 1e-12
    
    # 1. AMPLITUDE SHAPE FACTORS (4 features)
    features["form_factor"] = rms / mean_abs
    features["crest_factor"] = max_abs / rms
    features["impulse_factor"] = max_abs / mean_abs
    features["kurtosis"] = kurtosis(data)
    
    # 2. SIGNAL TEXTURE (3 features)
    # Zero crossing rate - normalized by time (Hz)
    features["zero_crossing_rate"] = np.count_nonzero(np.diff(np.signbit(data))) / (n / sr)
    
    # Hjorth mobility
    diff1 = np.diff(data)
    var_d0 = np.var(data)
    var_d1 = np.var(diff1)
    
    mobility = np.sqrt(var_d1 / var_d0) if var_d0 > 0 else 0
    features["hjorth_mobility"] = mobility
    
    # Skewness = measure of asymmetry
    features["skewness"] = skew(data)

    # 3. ONSET CHARACTERISTICS (3 features)
    for window_ms in [500, 1000]:
        samples = int((window_ms / 1000.0) * sr)
        if n > samples:
            max_window = np.max(np.abs(data[:samples]))
            features[f"onset_peak_ratio_{window_ms}ms"] = max_window / max_abs
        else:
            features[f"onset_peak_ratio_{window_ms}ms"] = 1.0
    
    samples_100ms = int(0.1 * sr)
    if n > samples_100ms:
        onset_segment = np.abs(data[:samples_100ms])
        if len(onset_segment) > 1:
            onset_gradient = np.gradient(onset_segment)
            features["onset_sharpness"] = np.max(onset_gradient) / (max_abs + 1e-12)
        else:
            features["onset_sharpness"] = 0
    else:
        features["onset_sharpness"] = 0

    # 4. FREQUENCY DOMAIN (16 features)
    freqs, fft, P = calculate_spectrum(data, sr, n)
    
    centroid = np.sum(freqs * P) / 20.0 if np.sum(P) > 0 else 0
    features["spectral_centroid_norm"] = centroid
    features["dominant_freq_norm"] = (freqs[np.argmax(fft)] / 20.0) if len(fft) > 0 else 0
    features["spectral_rolloff_85_norm"] = spectral_rolloff(freqs, fft, 0.85)
    features["spectral_rolloff_50_norm"] = spectral_rolloff(freqs, fft, 0.50)
    features["spectral_bandwidth_norm"] = features["spectral_rolloff_85_norm"] - features["spectral_rolloff_50_norm"]
    features["spectral_flatness"] = spectral_flatness(fft)
    
    if len(fft) > 0:
        sorted_fft = np.sort(fft)[::-1]
        if len(sorted_fft) > 1:
            features["peak_prominence"] = sorted_fft[0] / (sorted_fft[1] + 1e-12)
        else:
            features["peak_prominence"] = 1.0
    else:
        features["peak_prominence"] = 1.0
    
    # Energy distribution
    fft_sq = fft ** 2
    total_energy = np.sum(fft_sq) + 1e-12
    bands = [(0, 2), (2, 5), (5, 8), (8, 12), (12, 16), (16, 20)]
    for low, high in bands:
        idx = (freqs >= low) & (freqs < high)
        features[f"energy_ratio_{low}_{high}Hz"] = np.sum(fft_sq[idx]) / total_energy
    
    idx_low = freqs < 6
    idx_mid = (freqs >= 6) & (freqs < 12)
    idx_high = freqs >= 12
    
    energy_low = np.sum(fft_sq[idx_low])
    energy_mid = np.sum(fft_sq[idx_mid])
    energy_high = np.sum(fft_sq[idx_high])
    
    features["low_high_ratio"] = energy_low / (energy_high + 1e-12)
    features["mid_total_ratio"] = energy_mid / total_energy

    # 5. TEMPORAL EVOLUTION (3 features)
    quarters = np.array_split(data, 4)
    energy_quarters = [np.sum(q**2) for q in quarters]
    total_energy_time = np.sum(energy_quarters) + 1e-12
    
    features["energy_concentration"] = np.max(energy_quarters) / total_energy_time
    
    if len(energy_quarters) == 4:
        early_energy = energy_quarters[0] + energy_quarters[1]
        late_energy = energy_quarters[2] + energy_quarters[3]
        features["energy_decay_ratio"] = early_energy / (late_energy + 1e-12)
    else:
        features["energy_decay_ratio"] = 1.0
    
    centroids_q = []
    for q in quarters:
        if len(q) > 10:
            f_q, _, p_q = calculate_spectrum(q, sr, len(q))
            c_q = np.sum(f_q * p_q) / 20.0 if len(f_q) > 0 else 0
            centroids_q.append(c_q)
        else:
            centroids_q.append(0)
    
    if len(centroids_q) == 4 and centroids_q[0] > 0:
        features["freq_drift"] = (centroids_q[-1] - centroids_q[0]) / centroids_q[0]
    else:
        features["freq_drift"] = 0

    # 6. ENVELOPE FEATURES (8 features)
    env = np.abs(hilbert(data))
    k_size = int(sr * 0.1) | 1
    sm = signal.medfilt(env, kernel_size=max(3, k_size))
    mx_env = np.max(sm) + 1e-12
    sm_norm = sm / mx_env
    
    peak_idx = np.argmax(sm)
    features["peak_position_norm"] = peak_idx / n
    features["rise_decay_asymmetry"] = (peak_idx / n) / ((n - peak_idx) / n + 1e-12)
    
    for thresh in [0.2, 0.5]:
        above = np.where(sm_norm > thresh)[0]
        if len(above) > 0:
            features[f"duration_ratio_{int(thresh*100)}pct"] = (above[-1] - above[0]) / n
        else:
            features[f"duration_ratio_{int(thresh*100)}pct"] = 0
    
    features["envelope_skewness"] = skew(sm_norm)
    features["envelope_kurtosis"] = kurtosis(sm_norm)
    
    sm_norm_smooth = signal.medfilt(sm_norm, kernel_size=max(3, int(len(sm_norm) * 0.05) | 1))
    features["envelope_smoothness"] = np.corrcoef(sm_norm, sm_norm_smooth)[0, 1] if len(sm_norm) > 2 else 1.0
    
    peaks, _ = find_peaks(sm_norm, height=0.3, distance=int(sr * 0.2))
    features["envelope_peak_count_norm"] = len(peaks) / 10.0

    # 7. SIGNAL COMPLEXITY (1 feature)
    if len(data) > 1000:
        step = len(data) // 500
        data_sampled = data[::step]
    else:
        data_sampled = data
    
    if len(data_sampled) > 1:
        diff_std = np.std(np.diff(data_sampled))
        signal_std = np.std(data_sampled)
        features["signal_complexity"] = diff_std / (signal_std + 1e-12)
    else:
        features["signal_complexity"] = 0

    # 8. SPECTROGRAM FEATURES (4 features)
    spectrogram_feats = calculate_spectrogram_features(data, sr)
    features.update(spectrogram_feats)
    
    # 9. RECURRENCE FEATURES (2 features)
    recurrence_feats = calculate_recurrence_features(data, sr)
    features.update(recurrence_feats)
    
    # 10. WAVELET FEATURES (3 features)
    wavelet_feats = calculate_wavelet_features(data, sr)
    features.update(wavelet_feats)
    
    # 11. AUTOCORRELATION FEATURES (3 features)
    autocorr_feats = calculate_autocorr_features(data, sr)
    features.update(autocorr_feats)
    
    # 12. ADDITIONAL DISCRIMINATIVE FEATURES (2 features)
    # Signal entropy (dimensionless - normalized to max entropy)
    hist, _ = np.histogram(data, bins=50, density=True)
    hist = hist[hist > 0]
    if len(hist) > 0:
        features["signal_entropy_norm"] = -np.sum(hist * np.log2(hist + 1e-12)) / np.log2(50)
    else:
        features["signal_entropy_norm"] = 0
    
    # High-frequency content indicator
    if len(freqs) > 0:
        high_freq_idx = freqs > 10
        features["hf_content_ratio"] = np.sum(fft_sq[high_freq_idx]) / total_energy
    else:
        features["hf_content_ratio"] = 0

    return features

# =============== WORKER =================
def process_trigger(t):
    try:
        client = Client(FDSN_URL)
        
        t_start = t - window_before - buffer_time
        t_end   = t + window_after + buffer_time
        
        st = client.get_waveforms(network, station, "*", channel, t_start, t_end)
        if not st: return None

        st.detrend("linear")
        st.taper(0.05)
        st.filter("bandpass", freqmin=1, freqmax=20)
        st.trim(starttime=t - window_before, endtime=t + window_after)

        tr = st.select(component="Z")
        if not tr or len(tr) == 0 or len(tr[0].data) == 0: return None
        
        feats = extract_features(tr[0])
        feats["event_time"] = str(t)
        return feats
        
    except Exception as e:
        # Log exceptions to help diagnose hpc failures
        logger.error(f"{t}: {e}")
        return None

# =================== MAIN =================
def main():
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = pd.read_csv(input_file)
    times = [UTCDateTime(t) for t in df.iloc[:, 0]]
    total = len(times)

    logger.info(f"Starting feature extraction for {total} events")
    logger.info(f"Station: {network}.{station}, Channel: {channel}")
    logger.info(f"Window: {window_before}s before, {window_after}s after")
    logger.info(f"Workers: {MAX_WORKERS}")
    logger.info("-" * 70)
    
    results = []
    fails = 0
    start_time = time.time()
    last_log_time = start_time

    # ThreadPoolExecutor instead of ProcessPoolExecutor for HPC compatibility (important, remember)
    # FDSN Client uses network sockets which can deadlock with multiprocessing
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = [ex.submit(process_trigger, t) for t in times]

        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            if r:
                results.append(r)
            else:
                fails += 1
            
            current_time = time.time()
            if i % LOG_EVERY == 0 or i == total:
                elapsed = current_time - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                eta = format_time(remaining)
                success_rate = (len(results) / i) * 100 if i > 0 else 0
                
                logger.info(
                    f"Progress: {i}/{total} ({i/total*100:.1f}%) | "
                    f"Success: {len(results)} ({success_rate:.1f}%) | "
                    f"Failed: {fails} | "
                    f"Rate: {rate:.1f}/s | "
                    f"ETA: {eta}"
                )

    total_time = time.time() - start_time
    logger.info("-" * 70)
    logger.info(f"Extraction complete in {format_time(total_time)}")
    logger.info(f"Total processed: {len(results)}/{total} ({len(results)/total*100:.1f}%)")
    logger.info(f"Failed: {fails}")
    
    if results:
        df_out = pd.DataFrame(results)
        
        # Sort by event_time to maintain chronological order (important)
        df_out['event_time_dt'] = pd.to_datetime(df_out['event_time'])
        df_out = df_out.sort_values('event_time_dt').drop('event_time_dt', axis=1)
        
        # Organize columns: event_time first, then sorted features
        cols = ["event_time"] + sorted([c for c in df_out.columns if c != "event_time"])
        df_out = df_out[cols]
        
        # Log feature statistics
        logger.info(f"Features extracted: {len(cols)-1}")
        logger.info(f"Output dimensions: {df_out.shape}")
        
        df_out.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")
    else:
        logger.error("No features extracted. Check input data and network connection (especially Sophos).")

if __name__ == "__main__":
    main()