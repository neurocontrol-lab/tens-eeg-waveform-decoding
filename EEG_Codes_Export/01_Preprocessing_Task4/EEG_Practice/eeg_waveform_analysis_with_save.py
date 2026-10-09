"""
EEG Waveform Analysis Script - WITH SAVE/LOAD FUNCTIONALITY
Process TMSI SAGA 64-channel EEG data from XDF files
Analyzes different waveform stimuli at multiple amplitudes

NEW FEATURES:
- Save preprocessed data for reuse (save 40+ minutes!)
- Load preprocessed data instantly (5-10 seconds)
- Band info in plot titles
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from pyxdf import load_xdf
import os
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# Threshold values for each waveform type (modify these as needed)
SineThresh = 224
SquareThresh = 227
TriangleThresh = 230
LinIncThresh = 233
LinDecThresh = 236

Sine_T4 = 239
Square_T4 = 240
Triangle_T4 = 241
LinInc_T4 = 242
LinDec_T4 = 243

# Trigger mapping dictionary
#uncomment if task 3
TRIGGER_MAPPING = {
    # Sine wave
    'Sine_Base': SineThresh,
    'Sine_Mid': SineThresh + 1,  # Thresh + 0.25 amplitude
    'Sine_High': SineThresh + 2,  # Thresh + 0.5 amplitude
    
    # Square wave
    'Square_Base': SquareThresh,
    'Square_Mid': SquareThresh + 1,
    'Square_High': SquareThresh + 2,
    
    # Triangle wave
    'Triangle_Base': TriangleThresh,
    'Triangle_Mid': TriangleThresh + 1,
    'Triangle_High': TriangleThresh + 2,
    
    # Linear Increase
    'LinInc_Base': LinIncThresh,
    'LinInc_Mid': LinIncThresh + 1,
    'LinInc_High': LinIncThresh + 2,
    
    # Linear Decrease
    'LinDec_Base': LinDecThresh,
    'LinDec_Mid': LinDecThresh + 1,
    'LinDec_High': LinDecThresh + 2,
}
# Uncomment if task 4
# TRIGGER_MAPPING = {
#     'Sine': Sine_T4,
#     'Square': Square_T4,
#     'Triangle': Triangle_T4,
#     'LinInc': LinInc_T4,
#     'LinDec': LinDec_T4,
# }

# Waveform groupings T 3
WAVEFORM_GROUPS = {
    'Sine': [SineThresh, SineThresh + 1, SineThresh + 2],
    'Square': [SquareThresh, SquareThresh + 1, SquareThresh + 2],
    'Triangle': [TriangleThresh, TriangleThresh + 1, TriangleThresh + 2],
    'LinInc': [LinIncThresh, LinIncThresh + 1, LinIncThresh + 2],
    'LinDec': [LinDecThresh, LinDecThresh + 1, LinDecThresh + 2],
}
# Waveform groupings T 4
# WAVEFORM_GROUPS = {
#     'Sine': [Sine_T4],
#     'Square': [Square_T4],
#     'Triangle': [Triangle_T4],
#     'LinInc': [LinInc_T4],
#     'LinDec': [LinDec_T4],
# }

# Amplitude labels T 3
AMPLITUDE_LABELS = ['Base', 'Mid (+0.25)', 'High (+0.5)']

# Amplitude labels T 4
# AMPLITUDE_LABELS = ['Base']

# Preprocessing parameters
LOWCUT = 0.5  # Hz - high-pass filter
HIGHCUT = 40.0  # Hz - low-pass filter
NOTCH_FREQ = 60.0  # Hz - notch filter for line noise (use 50 for EU)

# Epoching parameters
TMIN = -0.2  # seconds before trigger
TMAX = 1.0   # seconds after trigger
BASELINE = (-0.2, 0)  # Baseline correction from start to trigger

# Predefined electrode groups for common regions of interest
ELECTRODE_GROUPS = {
    'central': ['Cz', 'C1', 'C2', 'C3', 'C4'],
    'frontal': ['Fz', 'F1', 'F2', 'F3', 'F4', 'F7', 'F8'],
    'parietal': ['Pz', 'P1', 'P2', 'P3', 'P4', 'P7', 'P8'],
    'occipital': ['Oz', 'O1', 'O2'],
    'temporal': ['T7', 'T8', 'TP7', 'TP8'],
    'frontocentral': ['Fz', 'F1', 'F2', 'Cz', 'C1', 'C2'],
    'centroparietal': ['Cz', 'C1', 'C2', 'Pz', 'P1', 'P2'],
    'midline': ['Fz', 'Cz', 'Pz', 'Oz'],
    'left_hemisphere': ['F3', 'F7', 'C3', 'T7', 'P3', 'P7', 'O1'],
    'right_hemisphere': ['F4', 'F8', 'C4', 'T8', 'P4', 'P8', 'O2'],
    'all': None  # None means use all available channels
}

# Frequency bands for oscillatory analysis
FREQUENCY_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100),
    'broadband': None  # No additional filtering (uses preprocessing filters)
}


# ==============================================================================
# FUNCTION DEFINITIONS
# ==============================================================================

def load_xdf_to_mne(xdf_filepath):
    """
    Load XDF file and convert to MNE Raw object
    TMSI SAGA specific: Triggers are embedded in the EEG stream as a separate channel
    
    Parameters:
    -----------
    xdf_filepath : str
        Path to the XDF file
        
    Returns:
    --------
    raw : mne.io.Raw
        MNE Raw object containing EEG data
    events : np.ndarray
        Events array (n_events, 3)
    """
    print(f"Loading XDF file: {xdf_filepath}")
    
    # Load XDF file
    streams, header = load_xdf(xdf_filepath)
    
    # Find TMSI SAGA stream (usually the first stream)
    saga_stream = None
    for stream in streams:
        # Look for the main data stream (typically SAGA)
        if 'time_series' in stream and stream['time_series'].shape[1] > 1:
            saga_stream = stream
            print(f"Found data stream: {stream['info']['name'][0]}")
            break
    
    if saga_stream is None:
        raise ValueError("No suitable data stream found in XDF file")
    
    # Parse channel information from XML descriptor
    channels = saga_stream['info']['desc'][0]['channels'][0]['channel']
    
    trigger_idx = None
    eeg_indices = []
    eeg_names = []
    
    print("\nParsing channels:")
    for idx, ch in enumerate(channels):
        ch_label = ch['label'][0]
        ch_type = ch['type'][0] if 'type' in ch else 'Unknown'
        
        if ch_label == 'TRIGGERS':
            trigger_idx = idx
            print(f"  Channel {idx}: {ch_label} (TRIGGER CHANNEL)")
        elif ch_type == 'EEG':  # Include all EEG channels, including CREF
            eeg_indices.append(idx)
            eeg_names.append(ch_label)
            if ch_label == 'CREF':
                print(f"  Channel {idx}: {ch_label} (EEG - Common Reference)")
            else:
                print(f"  Channel {idx}: {ch_label} (EEG)")
    
    if trigger_idx is None:
        raise ValueError("No TRIGGERS channel found in data stream")
    
    print(f"\nFound {len(eeg_names)} EEG channels and 1 trigger channel")
    
    # Extract EEG data and trigger data
    data = saga_stream['time_series'][:, eeg_indices].T  # (n_channels, n_samples)
    trigger_data = saga_stream['time_series'][:, trigger_idx]  # (n_samples,)
    sfreq = float(saga_stream['info']['nominal_srate'][0])
    
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Data shape: {data.shape} (channels × samples)")
    
    # Create MNE Info object
    info = mne.create_info(ch_names=eeg_names, sfreq=sfreq, ch_types='eeg')
    
    # Create Raw object
    raw = mne.io.RawArray(data, info)
    
    # Try to set montage automatically if channel names match standard 10-20
    print("\nAttempting to set electrode montage...")
    try:
        # Try standard 10-20 montage first
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')
        print("  Successfully set standard 10-20 montage")
        
        # Check how many channels were matched
        matched_channels = [ch for ch in raw.ch_names if ch in montage.ch_names]
        if len(matched_channels) > 0:
            print(f"  Matched {len(matched_channels)}/{len(raw.ch_names)} channels to standard positions")
        else:
            print("  Warning: No channels matched standard 10-20 names")
            print("  Topographic plots will not be available")
    except Exception as e:
        print(f"  Could not set montage automatically: {e}")
        print("  Topographic plots will not be available")
        print("  You can manually set montage if your channels follow a known layout")
    
    # Extract events from trigger channel
    # Find trigger onsets (when trigger value changes from 0 to non-zero)
    events = extract_triggers_from_channel(trigger_data, sfreq)
    
    print(f"Extracted {len(events)} trigger events")
    if len(events) > 0:
        unique_triggers = np.unique(events[:, 2]).astype(int)
        print(f"Unique trigger values: {unique_triggers}")
        for trig_val in unique_triggers:
            count = np.sum(events[:, 2] == trig_val)
            print(f"  Trigger {int(trig_val):3d}: {count} occurrences")
    
    return raw, events


def extract_triggers_from_channel(trigger_data, sfreq):
    """
    Extract trigger events from a continuous trigger channel
    Detects onsets (transitions from 0 or baseline to trigger value)
    
    Parameters:
    -----------
    trigger_data : np.ndarray
        Continuous trigger channel data (n_samples,)
    sfreq : float
        Sampling frequency
        
    Returns:
    --------
    events : np.ndarray
        Events array (n_events, 3) with [sample_idx, 0, trigger_value]
    """
    events_list = []
    
    # Method 1: Detect rising edges (transitions to non-zero values)
    # This handles triggers that go from 0 to a value
    prev_val = 0
    for i in range(len(trigger_data)):
        current_val = int(trigger_data[i])
        
        # Detect onset: transition from 0 (or different value) to non-zero
        if current_val != 0 and current_val != prev_val:
            events_list.append([i, 0, current_val])
        
        prev_val = current_val
    
    # Convert to numpy array
    if len(events_list) > 0:
        events = np.array(events_list, dtype=int)
    else:
        events = np.array([]).reshape(0, 3)
    
    return events


def preprocess_eeg(raw, l_freq=LOWCUT, h_freq=HIGHCUT, notch_freq=NOTCH_FREQ, 
                   apply_ica=True, n_components=15, plot_ica=True):
    """
    Preprocess EEG data with filtering, re-referencing, and ICA
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low-pass filter frequency (high-pass cutoff)
    h_freq : float
        High-pass filter frequency (low-pass cutoff)
    notch_freq : float
        Notch filter frequency for line noise
    apply_ica : bool
        Whether to apply ICA for artifact removal
    n_components : int
        Number of ICA components to compute
    plot_ica : bool
        Whether to plot ICA components for inspection
        
    Returns:
    --------
    raw_preprocessed : mne.io.Raw
        Preprocessed EEG data
    ica : mne.preprocessing.ICA or None
        Fitted ICA object (if apply_ica=True)
    """
    print("\n" + "="*80)
    print("PREPROCESSING EEG DATA")
    print("="*80)
    
    # Make a copy to avoid modifying original
    raw_preprocessed = raw.copy()
    
    # Apply bandpass filter
    print(f"Applying bandpass filter: {l_freq}-{h_freq} Hz")
    raw_preprocessed.filter(l_freq, h_freq, fir_design='firwin')
    
    # Apply notch filter for line noise
    print(f"Applying notch filter at {notch_freq} Hz")
    raw_preprocessed.notch_filter(notch_freq, fir_design='firwin')
    
    # Apply average reference
    # Apply linked mastoid reference (M1 + M2)
    print("Applying linked mastoid reference (M1 + M2)")

    # Check that M1 and M2 exist in the data
    mastoid_channels = ['M1', 'M2']
    available_mastoids = [ch for ch in mastoid_channels if ch in raw_preprocessed.ch_names]

    if len(available_mastoids) == 2:
        # Set linked mastoid reference using both M1 and M2
        raw_preprocessed.set_eeg_reference(ref_channels=['M1', 'M2'], projection=False)
        # raw_preprocessed.apply_proj()
        print(f"  Successfully applied linked mastoid reference (M1 + M2)")
    elif len(available_mastoids) == 1:
        # Fall back to single mastoid if only one is available
        print(f"  Warning: Only {available_mastoids[0]} found. Using single mastoid reference.")
        raw_preprocessed.set_eeg_reference(ref_channels=available_mastoids, projection=False)
        # raw_preprocessed.apply_proj()
    else:
        # Fall back to average reference if mastoids not found
        print(f"  Warning: M1/M2 channels not found in data. Falling back to average reference.")
        print(f"  Available channels: {raw_preprocessed.ch_names}")
        raw_preprocessed.set_eeg_reference('average', projection=True)
        raw_preprocessed.apply_proj()
    
    # Apply ICA
    ica = None
    if apply_ica:
        print("\n" + "-"*80)
        print("INDEPENDENT COMPONENT ANALYSIS (ICA)")
        print("-"*80)
        
        # Create a copy for ICA (use higher high-pass for better ICA results)
        print("Creating temporary copy with 1 Hz high-pass filter for ICA...")
        raw_ica = raw_preprocessed.copy()
        raw_ica.filter(1.0, None, fir_design='firwin')
        
        # Fit ICA
        print(f"Fitting ICA with {n_components} components...")
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=42,
            max_iter='auto',
            method='infomax',
            fit_params=dict(extended=True)
        )
        
        ica.fit(raw_ica)
        print(f"ICA fitting complete. Explained variance: {ica.get_explained_variance_ratio(raw_ica)['eeg']:.2%}")
        
        # Plot ICA components if requested
        if plot_ica:
            print("\nGenerating ICA component plots...")
            
            # Create a text report regardless of montage availability
            ica_report_path = 'ica_components_report.txt'
            with open(ica_report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("ICA COMPONENTS ANALYSIS REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(f"Number of components: {ica.n_components_}\n")
                f.write(f"Explained variance: {ica.get_explained_variance_ratio(raw_ica)['eeg']:.2%}\n")
                f.write(f"Components excluded: {sorted(ica.exclude)}\n\n")
                
                f.write("-"*80 + "\n")
                f.write("COMPONENT DETAILS\n")
                f.write("-"*80 + "\n\n")
                
                for i in range(ica.n_components_):
                    status = "EXCLUDED" if i in ica.exclude else "included"
                    f.write(f"Component {i:02d}: {status}\n")
                
                if len(ica.exclude) > 0:
                    f.write("\n" + "-"*80 + "\n")
                    f.write("EXCLUDED COMPONENTS (artifacts)\n")
                    f.write("-"*80 + "\n")
                    for i in sorted(ica.exclude):
                        f.write(f"  Component {i:02d}\n")
                
            print(f"  Saved: {ica_report_path}")
            
            try:
                # Try to plot ICA components (requires electrode positions)
                fig = ica.plot_components(show=False)
                plt.savefig('ica_components_topomap.png', dpi=150, bbox_inches='tight')
                print("  Saved: ica_components_topomap.png")
                plt.close(fig)
            except RuntimeError as e:
                if "No digitization points found" in str(e) or "No electrode locations" in str(e):
                    print("  Warning: Cannot create topographic maps (no electrode positions)")
                    print("  Skipping component topomap visualization")
                    print("  Tip: Set a montage in load_xdf_to_mne() to enable topographic plots")
                else:
                    raise
            
            try:
                # Plot ICA sources (time series) - this doesn't require positions
                fig = ica.plot_sources(raw_ica, show=False, start=0, stop=10)
                plt.savefig('ica_sources_timeseries.png', dpi=150, bbox_inches='tight')
                print("  Saved: ica_sources_timeseries.png")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: Could not plot ICA sources: {e}")
            
            # Close any remaining figures
            plt.close('all')
        
        # Automatic component rejection based on common criteria
        print("\nSearching for artifact components...")
        
        # Find blink artifacts (if EOG channels available, otherwise use automatic detection)
        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw_ica, threshold=3.0)
            if len(eog_indices) > 0:
                print(f"  Found {len(eog_indices)} potential EOG/blink components: {eog_indices}")
                ica.exclude.extend(eog_indices)
        except:
            print("  No EOG detection performed (no EOG channels or insufficient data)")
        
        # Find muscle artifacts
        try:
            muscle_idx_auto, muscle_scores = ica.find_bads_muscle(raw_ica)
            if len(muscle_idx_auto) > 0:
                print(f"  Found {len(muscle_idx_auto)} potential muscle artifact components: {muscle_idx_auto}")
                ica.exclude.extend(muscle_idx_auto)
        except:
            print("  No muscle artifact detection performed")
        
        # Remove duplicates from exclude list
        ica.exclude = list(set(ica.exclude))
        
        if len(ica.exclude) > 0:
            print(f"\nComponents marked for exclusion: {sorted(ica.exclude)}")
            
            # Plot excluded components
            if plot_ica and len(ica.exclude) > 0:
                try:
                    fig = ica.plot_properties(raw_ica, picks=ica.exclude, show=False)
                    plt.savefig('ica_excluded_components.png', dpi=150, bbox_inches='tight')
                    print("  Saved detailed view: ica_excluded_components.png")
                    plt.close('all')
                except RuntimeError as e:
                    if "No digitization points found" in str(e) or "No electrode locations" in str(e):
                        print("  Note: Cannot plot component properties (no electrode positions)")
                    else:
                        print(f"  Warning: Could not plot excluded components: {e}")
                except Exception as e:
                    print(f"  Warning: Could not plot excluded components: {e}")
            
            # Apply ICA to remove artifacts
            print(f"\nApplying ICA to remove {len(ica.exclude)} components...")
            ica.apply(raw_preprocessed)
            print("  Artifact removal complete!")
        else:
            print("\nNo components automatically marked for exclusion.")
            print("  You can manually inspect components and exclude them if needed.")
    
    print("\n" + "-"*80)
    print("Preprocessing complete!")
    print("="*80)
    
    return raw_preprocessed, ica


def apply_frequency_band(raw, band_name='broadband'):
    """
    Apply frequency band filtering to extract specific oscillatory activity
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Preprocessed EEG data
    band_name : str
        Name of frequency band to extract
        
    Returns:
    --------
    raw_band : mne.io.Raw
        Band-filtered EEG data
    band_info : dict
        Dictionary with band information (name, range, label)
    """
    if band_name not in FREQUENCY_BANDS:
        print(f"Warning: Unknown frequency band '{band_name}'")
        print(f"Available bands: {list(FREQUENCY_BANDS.keys())}")
        print("Using broadband (no additional filtering)")
        band_name = 'broadband'
    
    band_range = FREQUENCY_BANDS[band_name]
    
    if band_range is None:
        # Broadband - no additional filtering
        print(f"\nUsing broadband data (no frequency band filtering)")
        return raw.copy(), {'name': 'broadband', 'range': None, 'label': 'Broadband'}
    else:
        # Apply bandpass filter for specific band
        low_freq, high_freq = band_range
        print(f"\n" + "="*80)
        print(f"EXTRACTING {band_name.upper()} BAND ({low_freq}-{high_freq} Hz)")
        print("="*80)
        
        raw_band = raw.copy()
        raw_band.filter(low_freq, high_freq, fir_design='firwin')
        
        print(f"Applied {low_freq}-{high_freq} Hz bandpass filter")
        print(f"Data now contains {band_name} oscillatory activity")
        
        band_info = {
            'name': band_name,
            'range': (low_freq, high_freq),
            'label': f'{band_name.capitalize()} ({low_freq}-{high_freq} Hz)'
        }
        
        return raw_band, band_info


def create_epochs(raw, events, event_dict, tmin=TMIN, tmax=TMAX, baseline=BASELINE):
    """
    Create epochs based on events
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Preprocessed EEG data
    events : np.ndarray
        Events array
    event_dict : dict
        Dictionary mapping event names to event IDs
    tmin : float
        Start time before event
    tmax : float
        End time after event
    baseline : tuple
        Baseline correction period
        
    Returns:
    --------
    epochs : mne.Epochs
        Epoched EEG data
    """
    print("\n" + "="*80)
    print("CREATING EPOCHS")
    print("="*80)
    
    # Create epochs
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=event_dict,
        tmin=tmin, 
        tmax=tmax,
        baseline=baseline,
        preload=True,
        proj=True,
        reject_by_annotation=True
    )
    
    print(f"Created {len(epochs)} epochs")
    print(f"Epoch time range: {tmin} to {tmax} seconds")
    print(f"\nEpoch counts by condition:")
    for condition, count in epochs.event_id.items():
        n_epochs = len(epochs[condition])
        print(f"  {condition}: {n_epochs} epochs")
    
    return epochs


# ==============================================================================
# COMBINE MULTIPLE SUBJECTS - GRAND AVERAGE (NEW!)
# ==============================================================================

def load_and_combine_subjects(output_dir, subject_session_list, combine_sessions_per_subject=True):
    """
    Load and combine data across multiple subjects for grand average analysis
    
    This creates a grand average across subjects, pooling data from multiple
    participants. Each subject can contribute one or more sessions.
    
    Parameters:
    -----------
    output_dir : str
        Base directory containing saved files
    subject_session_list : list of tuples or list of lists
        Either:
        - List of (subject_id, session_id) tuples for single session per subject
        - List of lists: [[subject1_sessions], [subject2_sessions], ...]
        
        Examples:
        # Single session per subject
        [(1, 'Task1'), (2, 'Task1'), (3, 'Task1')]
        
        # Multiple sessions per subject (will be combined within subject first)
        [
            [(1, 'Task1'), (1, 'Task2')],  # Subject 1: 2 sessions
            [(2, 'Task1'), (2, 'Task2')],  # Subject 2: 2 sessions
            [(3, 'Task1'), (3, 'Task2')]   # Subject 3: 2 sessions
        ]
    
    combine_sessions_per_subject : bool
        If True and multiple sessions per subject, combine them first (default: True)
        If False, treat each session as independent
    
    Returns:
    --------
    raw_grand_avg : mne.io.Raw
        Combined raw data across subjects
    epochs_grand_avg : mne.Epochs
        Combined epochs across subjects
    subject_evokeds : dict
        Dictionary mapping subject_id to their evoked responses
        Format: {subject_id: {condition: evoked}}
    
    Example:
    --------
    # Grand average from 3 subjects, 1 session each
    subjects = [(1, 'Task1'), (2, 'Task1'), (3, 'Task1')]
    raw, epochs, evokeds = load_and_combine_subjects('./preprocessed', subjects)
    
    # Grand average from 3 subjects, 2 sessions each
    subjects = [
        [(1, 'Task1'), (1, 'Task2')],
        [(2, 'Task1'), (2, 'Task2')],
        [(3, 'Task1'), (3, 'Task2')]
    ]
    raw, epochs, evokeds = load_and_combine_subjects('./preprocessed', subjects)
    """
    print("\n" + "="*80)
    print("LOADING AND COMBINING MULTIPLE SUBJECTS - GRAND AVERAGE")
    print("="*80)
    
    # Parse input to determine structure
    if all(isinstance(item, tuple) for item in subject_session_list):
        # Single session per subject
        subject_groups = [[item] for item in subject_session_list]
        print(f"Mode: Single session per subject")
        print(f"Number of subjects: {len(subject_groups)}")
    elif all(isinstance(item, list) for item in subject_session_list):
        # Multiple sessions per subject
        subject_groups = subject_session_list
        print(f"Mode: Multiple sessions per subject")
        print(f"Number of subjects: {len(subject_groups)}")
    else:
        raise ValueError("Invalid subject_session_list format")
    
    raw_list = []
    epochs_list = []
    subject_evokeds = {}
    
    # Process each subject
    for subject_idx, subject_sessions in enumerate(subject_groups):
        print(f"\n{'='*60}")
        print(f"Processing Subject {subject_idx + 1}/{len(subject_groups)}")
        print('='*60)
        
        if len(subject_sessions) == 1:
            # Single session for this subject
            subject_id, session_id = subject_sessions[0]
            print(f"Subject ID: {subject_id}, Session: {session_id}")
            
            try:
                raw, epochs, ica = load_preprocessed_data(output_dir, subject_id, session_id)
                raw_list.append(raw)
                epochs_list.append(epochs)
                
                # Store evoked for this subject
                subject_evokeds[subject_id] = {}
                for condition in epochs.event_id.keys():
                    subject_evokeds[subject_id][condition] = epochs[condition].average()
                
                print(f"  ✓ Loaded {len(epochs)} epochs")
                
            except FileNotFoundError as e:
                print(f"  ✗ Warning: Could not load subject - {e}")
                continue
        
        else:
            # Multiple sessions for this subject
            subject_id = subject_sessions[0][0]  # Get subject ID from first session
            print(f"Subject ID: {subject_id}, Sessions: {len(subject_sessions)}")
            
            if combine_sessions_per_subject:
                # Combine sessions within subject first
                try:
                    raw_subj, epochs_subj, icas = load_and_combine_preprocessed_data(
                        output_dir, subject_sessions
                    )
                    raw_list.append(raw_subj)
                    epochs_list.append(epochs_subj)
                    
                    # Store evoked for this subject
                    subject_evokeds[subject_id] = {}
                    for condition in epochs_subj.event_id.keys():
                        subject_evokeds[subject_id][condition] = epochs_subj[condition].average()
                    
                    print(f"  ✓ Combined {len(subject_sessions)} sessions: {len(epochs_subj)} total epochs")
                    
                except FileNotFoundError as e:
                    print(f"  ✗ Warning: Could not load subject - {e}")
                    continue
            else:
                # Treat each session independently
                for subject_id, session_id in subject_sessions:
                    try:
                        raw, epochs, ica = load_preprocessed_data(output_dir, subject_id, session_id)
                        raw_list.append(raw)
                        epochs_list.append(epochs)
                        print(f"  ✓ Loaded session {session_id}: {len(epochs)} epochs")
                    except FileNotFoundError as e:
                        print(f"  ✗ Warning: Could not load session {session_id} - {e}")
                        continue
    
    if len(epochs_list) == 0:
        raise ValueError("No valid subjects found to combine!")
    
    print("\n" + "="*80)
    print("COMBINING SUBJECTS - CREATING GRAND AVERAGE")
    print("="*80)
    
    # Concatenate raw data
    if len(raw_list) > 1:
        print("Concatenating raw data across subjects...")
        raw_grand_avg = mne.concatenate_raws(raw_list, preload=True)
        print(f"  Combined raw data shape: {raw_grand_avg.get_data().shape}")
    else:
        raw_grand_avg = raw_list[0]
    
    # Concatenate epochs
    if len(epochs_list) > 1:
        print("\nConcatenating epochs across subjects...")
        epochs_grand_avg = mne.concatenate_epochs(epochs_list)
        print(f"  Grand average total epochs: {len(epochs_grand_avg)}")
        
        # Show grand average epoch counts by condition
        print("\nGrand average epoch counts by condition:")
        for condition in epochs_grand_avg.event_id.keys():
            n_epochs = len(epochs_grand_avg[condition])
            print(f"  {condition}: {n_epochs} epochs")
    else:
        epochs_grand_avg = epochs_list[0]
    
    print("\n" + "="*80)
    print("GRAND AVERAGE COMPLETE!")
    print("="*80)
    print(f"Total subjects: {len(subject_evokeds)}")
    print(f"Total epochs: {len(epochs_grand_avg)}")
    print(f"Sampling rate: {epochs_grand_avg.info['sfreq']} Hz")
    
    return raw_grand_avg, epochs_grand_avg, subject_evokeds


def grand_average_analysis(output_dir, subject_session_list,
                           electrode_groups=None, 
                           frequency_band='broadband',
                           output_plots_dir=None,
                           combine_sessions_per_subject=True):
    """
    Complete grand average workflow: Load multiple subjects, combine, and create plots
    
    This function creates group-level plots showing the average response across
    all subjects. Confidence intervals now represent between-subject variability.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing preprocessed data
    subject_session_list : list
        List of subjects/sessions (see load_and_combine_subjects for format)
    electrode_groups : dict, list, str, or None
        Electrodes to plot
    frequency_band : str
        Frequency band to extract (default: 'broadband')
    output_plots_dir : str, optional
        Directory to save plots. If None, uses '{output_dir}/grand_average_plots'
    combine_sessions_per_subject : bool
        Whether to combine sessions within each subject first (default: True)
    
    Returns:
    --------
    raw_grand_avg : mne.io.Raw
        Grand average raw data
    epochs_grand_avg : mne.Epochs
        Grand average epochs
    subject_evokeds : dict
        Individual subject evoked responses
    
    Example:
    --------
    # Grand average from 3 subjects
    subjects = [(1, 'Task1'), (2, 'Task1'), (3, 'Task1')]
    
    raw, epochs, evokeds = grand_average_analysis(
        './preprocessed_data',
        subjects,
        electrode_groups='central',
        frequency_band='alpha',
        output_plots_dir='./grand_average_alpha'
    )
    """
    # Set output directory
    if output_plots_dir is None:
        output_plots_dir = os.path.join(output_dir, 'grand_average_plots')
    os.makedirs(output_plots_dir, exist_ok=True)
    
    # Load and combine subjects
    raw_grand_avg, epochs_grand_avg, subject_evokeds = load_and_combine_subjects(
        output_dir, subject_session_list, combine_sessions_per_subject
    )
    
    # Apply frequency band filtering
    raw_band, band_info = apply_frequency_band(raw_grand_avg, frequency_band)
    
    # Create epochs from band-filtered data
    print("\nCreating epochs from band-filtered data...")
    epochs_band = create_epochs(raw_band, epochs_grand_avg.events, TRIGGER_MAPPING)
    
    # Process electrode groups
    if electrode_groups is None:
        plot_groups = {'all': None}
    elif isinstance(electrode_groups, str):
        if electrode_groups in ELECTRODE_GROUPS:
            plot_groups = {electrode_groups: ELECTRODE_GROUPS[electrode_groups]}
        else:
            print(f"Warning: Unknown electrode group '{electrode_groups}'")
            print(f"Available groups: {list(ELECTRODE_GROUPS.keys())}")
            print("Using all electrodes instead.")
            plot_groups = {'all': None}
    elif isinstance(electrode_groups, list):
        plot_groups = {'custom': electrode_groups}
    elif isinstance(electrode_groups, dict):
        plot_groups = electrode_groups
    else:
        print("Warning: Invalid electrode_groups format. Using all electrodes.")
        plot_groups = {'all': None}
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING GRAND AVERAGE PLOTS")
    print("="*80)
    print(f"Subjects: {len(subject_evokeds)}")
    print(f"Total epochs: {len(epochs_band)}")
    
    for group_name, electrodes in plot_groups.items():
        print(f"\nPlotting electrode group: {group_name}")
        if band_info['name'] != 'broadband':
            print(f"Frequency band: {band_info['label']}")
        print("-" * 60)
        
        # Create subdirectory for this electrode group if multiple groups
        if len(plot_groups) > 1:
            group_output_dir = os.path.join(output_plots_dir, f'group_{group_name}')
            os.makedirs(group_output_dir, exist_ok=True)
        else:
            group_output_dir = output_plots_dir
        
        # Create filename suffix
        electrode_suffix = f"_{group_name}" if electrodes is not None and group_name != 'all' else ""
        band_suffix = f"_{band_info['name']}_{band_info['range'][0]}-{band_info['range'][1]}Hz" if band_info['name'] != 'broadband' else ""
        subjects_suffix = f"_grandavg{len(subject_evokeds)}subjects"
        
        # Plot individual waveforms
        for waveform_name, trigger_values in WAVEFORM_GROUPS.items():
            filename = f'{waveform_name}_evoked{electrode_suffix}{band_suffix}{subjects_suffix}.png'
            save_path = os.path.join(group_output_dir, filename)
            plot_waveform_evoked(epochs_band, waveform_name, trigger_values, 
                               electrodes=electrodes, band_info=band_info,
                               save_path=save_path)
        
        # Create comparison plot
        filename = f'all_waveforms_comparison{electrode_suffix}{band_suffix}{subjects_suffix}.png'
        save_path = os.path.join(group_output_dir, filename)
        plot_all_waveforms_comparison(epochs_band, electrodes=electrodes,
                                     band_info=band_info, save_path=save_path)
        
        
        filename = f'all_amplitude_comparison{electrode_suffix}{band_suffix}{subjects_suffix}.png'
        save_path = os.path.join(group_output_dir, filename)
        plot_amplitude_comparison(epochs_band, electrodes=electrodes,
                                    band_info=band_info, save_path=save_path)
    
    print("\n" + "="*80)
    print("GRAND AVERAGE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Plots saved to: {output_plots_dir}")
    
    return raw_grand_avg, epochs_band, subject_evokeds


# ==============================================================================
# COMBINE MULTIPLE PREPROCESSED DATASETS (NEW!)
# ==============================================================================

def load_and_combine_preprocessed_data(output_dir, session_list):
    """
    Load and combine preprocessed data from multiple sessions/trials
    
    This allows you to average epochs across multiple recording sessions
    to increase statistical power and reduce noise.
    
    Parameters:
    -----------
    output_dir : str
        Base directory containing saved files
    session_list : list of tuples
        List of (subject_id, session_id) tuples to combine
        Example: [(3, 'Task1'), (3, 'Task2'), (3, 'Task3')]
    
    Returns:
    --------
    raw_combined : mne.io.Raw
        Combined raw data (concatenated)
    epochs_combined : mne.Epochs
        Combined epochs (concatenated across sessions)
    ica_list : list
        List of ICA solutions from each session
    
    Example:
    --------
    # Combine data from 3 sessions for subject 3
    sessions = [(3, 'Task1'), (3, 'Task2'), (3, 'Task3')]
    raw, epochs, icas = load_and_combine_preprocessed_data('./preprocessed', sessions)
    """
    print("\n" + "="*80)
    print("LOADING AND COMBINING MULTIPLE PREPROCESSED DATASETS")
    print("="*80)
    print(f"Number of sessions to combine: {len(session_list)}")
    
    raw_list = []
    epochs_list = []
    ica_list = []
    
    # Load each session
    for subject_id, session_id in session_list:
        print(f"\nLoading session: Subject {subject_id}, {session_id}")
        print("-" * 60)
        
        try:
            raw, epochs, ica = load_preprocessed_data(output_dir, subject_id, session_id)
            raw_list.append(raw)
            epochs_list.append(epochs)
            ica_list.append(ica)
            print(f"  ✓ Loaded {len(epochs)} epochs")
        except FileNotFoundError as e:
            print(f"  ✗ Warning: Could not load session - {e}")
            continue
    
    if len(epochs_list) == 0:
        raise ValueError("No valid sessions found to combine!")
    
    print("\n" + "="*80)
    print("COMBINING DATA")
    print("="*80)
    
    # Concatenate raw data
    if len(raw_list) > 1:
        print("Concatenating raw data...")
        raw_combined = mne.concatenate_raws(raw_list, preload=True)
        print(f"  Combined raw data shape: {raw_combined.get_data().shape}")
    else:
        raw_combined = raw_list[0]
    
    # Concatenate epochs
    if len(epochs_list) > 1:
        print("\nConcatenating epochs...")
        epochs_combined = mne.concatenate_epochs(epochs_list)
        print(f"  Combined total epochs: {len(epochs_combined)}")
        
        # Show combined epoch counts by condition
        print("\nCombined epoch counts by condition:")
        for condition in epochs_combined.event_id.keys():
            n_epochs = len(epochs_combined[condition])
            print(f"  {condition}: {n_epochs} epochs")
    else:
        epochs_combined = epochs_list[0]
    
    print("\n" + "="*80)
    print("COMBINING COMPLETE!")
    print("="*80)
    print(f"Total sessions combined: {len(session_list)}")
    print(f"Total epochs: {len(epochs_combined)}")
    print(f"Sampling rate: {epochs_combined.info['sfreq']} Hz")
    
    return raw_combined, epochs_combined, ica_list


def combine_and_analyze_multiple_sessions(output_dir, session_list, 
                                          electrode_groups=None, 
                                          frequency_band='broadband',
                                          output_plots_dir=None):
    """
    Complete workflow: Load multiple sessions, combine, and create plots
    
    Parameters:
    -----------
    output_dir : str
        Directory containing preprocessed data
    session_list : list of tuples
        List of (subject_id, session_id) tuples
        Example: [(3, 'Task1'), (3, 'Task2'), (3, 'Task3')]
    electrode_groups : dict, list, str, or None
        Electrodes to plot (same as analyze_eeg_waveforms)
    frequency_band : str
        Frequency band to extract (default: 'broadband')
    output_plots_dir : str, optional
        Directory to save plots. If None, uses '{output_dir}/combined_plots'
    
    Returns:
    --------
    raw_combined : mne.io.Raw
        Combined raw data
    epochs_combined : mne.Epochs
        Combined epochs  
    ica_list : list
        List of ICA solutions
    
    Example:
    --------
    # Combine 3 sessions and plot central electrodes, alpha band
    sessions = [(3, 'Task1'), (3, 'Task2'), (3, 'Task3')]
    raw, epochs, icas = combine_and_analyze_multiple_sessions(
        './preprocessed',
        sessions,
        electrode_groups='central',
        frequency_band='alpha',
        output_plots_dir='./combined_analysis'
    )
    """
    # Set output directory
    if output_plots_dir is None:
        output_plots_dir = os.path.join(output_dir, 'combined_plots')
    os.makedirs(output_plots_dir, exist_ok=True)
    
    # Load and combine sessions
    raw_combined, epochs_combined, ica_list = load_and_combine_preprocessed_data(
        output_dir, session_list
    )
    
    # Apply frequency band filtering to combined data
    raw_band, band_info = apply_frequency_band(raw_combined, frequency_band)
    
    # Create epochs from band-filtered data
    print("\nCreating epochs from band-filtered data...")
    epochs_band = create_epochs(raw_band, epochs_combined.events, TRIGGER_MAPPING)
    
    # Process electrode groups (same as main analysis function)
    if electrode_groups is None:
        plot_groups = {'all': None}
    elif isinstance(electrode_groups, str):
        if electrode_groups in ELECTRODE_GROUPS:
            plot_groups = {electrode_groups: ELECTRODE_GROUPS[electrode_groups]}
        else:
            print(f"Warning: Unknown electrode group '{electrode_groups}'")
            print(f"Available groups: {list(ELECTRODE_GROUPS.keys())}")
            print("Using all electrodes instead.")
            plot_groups = {'all': None}
    elif isinstance(electrode_groups, list):
        plot_groups = {'custom': electrode_groups}
    elif isinstance(electrode_groups, dict):
        plot_groups = electrode_groups
    else:
        print("Warning: Invalid electrode_groups format. Using all electrodes.")
        plot_groups = {'all': None}
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING COMBINED PLOTS")
    print("="*80)
    print(f"Sessions combined: {len(session_list)}")
    print(f"Total epochs: {len(epochs_band)}")
    
    for group_name, electrodes in plot_groups.items():
        print(f"\nPlotting electrode group: {group_name}")
        if band_info['name'] != 'broadband':
            print(f"Frequency band: {band_info['label']}")
        print("-" * 60)
        
        # Create subdirectory for this electrode group if multiple groups
        if len(plot_groups) > 1:
            group_output_dir = os.path.join(output_plots_dir, f'group_{group_name}')
            os.makedirs(group_output_dir, exist_ok=True)
        else:
            group_output_dir = output_plots_dir
        
        # Create filename suffix
        electrode_suffix = f"_{group_name}" if electrodes is not None and group_name != 'all' else ""
        band_suffix = f"_{band_info['name']}_{band_info['range'][0]}-{band_info['range'][1]}Hz" if band_info['name'] != 'broadband' else ""
        sessions_suffix = f"_combined{len(session_list)}sessions"
        
        # Plot individual waveforms
        for waveform_name, trigger_values in WAVEFORM_GROUPS.items():
            filename = f'{waveform_name}_evoked{electrode_suffix}{band_suffix}{sessions_suffix}.png'
            save_path = os.path.join(group_output_dir, filename)
            plot_waveform_evoked(epochs_band, waveform_name, trigger_values, 
                               electrodes=electrodes, band_info=band_info,
                               save_path=save_path)
        
        # Create comparison plot
        filename = f'all_waveforms_comparison{electrode_suffix}{band_suffix}{sessions_suffix}.png'
        save_path = os.path.join(group_output_dir, filename)
        plot_all_waveforms_comparison(epochs_band, electrodes=electrodes,
                                     band_info=band_info, save_path=save_path)
    
    print("\n" + "="*80)
    print("COMBINED ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Plots saved to: {output_plots_dir}")
    
    return raw_combined, epochs_band, ica_list


# ==============================================================================
# SAVE/LOAD PREPROCESSED DATA FUNCTIONS (NEW!)
# ==============================================================================

def save_preprocessed_data(raw, epochs, ica, output_dir, subject_id, session_id):
    """
    Save preprocessed data using MNE's native format
    
    This allows you to load the preprocessed data later without redoing preprocessing
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Preprocessed raw data
    epochs : mne.Epochs
        Epoched data
    ica : mne.preprocessing.ICA or None
        ICA solution
    output_dir : str
        Directory to save files
    subject_id : str or int
        Subject identifier
    session_id : str
        Session identifier (e.g., 'Task3')
    
    Returns:
    --------
    saved_files : dict
        Dictionary with paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING PREPROCESSED DATA")
    print("="*80)
    
    saved_files = {}
    
    # Save raw data
    raw_fname = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_raw-preprocessed.fif')
    raw.save(raw_fname, overwrite=True)
    saved_files['raw'] = raw_fname
    print(f"Saved preprocessed raw data: {raw_fname}")
    
    # Save epochs
    epochs_fname = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_epo.fif')
    epochs.save(epochs_fname, overwrite=True)
    saved_files['epochs'] = epochs_fname
    print(f"Saved epochs: {epochs_fname}")
    
    # Save ICA
    if ica is not None:
        ica_fname = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_ica.fif')
        ica.save(ica_fname, overwrite=True)
        saved_files['ica'] = ica_fname
        print(f"Saved ICA solution: {ica_fname}")
    
    print("\n" + "="*80)
    print("SAVE COMPLETE")
    print("="*80)
    print("\nTo load later:")
    print(f"  raw = mne.io.read_raw_fif('{raw_fname}')")
    print(f"  epochs = mne.read_epochs('{epochs_fname}')")
    if ica is not None:
        print(f"  ica = mne.preprocessing.read_ica('{ica_fname}')")
    
    return saved_files


def load_preprocessed_data(output_dir, subject_id, session_id):
    """
    Load previously saved preprocessed data
    
    Parameters:
    -----------
    output_dir : str
        Directory containing saved files
    subject_id : str or int
        Subject identifier
    session_id : str
        Session identifier
    
    Returns:
    --------
    raw : mne.io.Raw
        Preprocessed raw data
    epochs : mne.Epochs
        Epoched data
    ica : mne.preprocessing.ICA or None
        ICA solution
    """
    print("\n" + "="*80)
    print("LOADING PREPROCESSED DATA")
    print("="*80)
    
    # Load raw
    raw_fname = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_raw-preprocessed.fif')
    if os.path.exists(raw_fname):
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        print(f"Loaded raw data: {raw_fname}")
    else:
        raise FileNotFoundError(f"Raw data file not found: {raw_fname}")
    
    # Load epochs
    epochs_fname = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_epo.fif')
    if os.path.exists(epochs_fname):
        epochs = mne.read_epochs(epochs_fname)
        print(f"Loaded epochs: {epochs_fname}")
    else:
        raise FileNotFoundError(f"Epochs file not found: {epochs_fname}")
    
    # Load ICA
    ica_fname = os.path.join(output_dir, f'sub-{subject_id}_ses-{session_id}_ica.fif')
    if os.path.exists(ica_fname):
        ica = mne.preprocessing.read_ica(ica_fname)
        print(f"Loaded ICA: {ica_fname}")
    else:
        print("No ICA file found")
        ica = None
    
    print("="*80)
    
    return raw, epochs, ica


# ==============================================================================
# PLOTTING FUNCTIONS (UPDATED WITH BAND INFO)
# ==============================================================================

def plot_waveform_evoked(epochs, waveform_name, trigger_values, electrodes=None, 
                        band_info=None, save_path=None, ci=0.90):
    """
    Plot evoked responses using MNE's native plotting with confidence intervals
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched data
    waveform_name : str
        Name of the waveform (e.g., 'Sine', 'Square')
    trigger_values : list
        List of trigger values for this waveform [base, mid, high]
    electrodes : list of str or None
        List of electrode names to include in the plot
    band_info : dict or None
        Dictionary with band information (for plot titles)
    save_path : str, optional
        Path to save the figure
    ci : float, optional
        Confidence interval level (default: 0.90 for 90% CI)
        Options: 0.90 (90%), 0.95 (95%), 0.99 (99%)
    """
    print(f"\nPlotting {waveform_name} waveform...")
    
    # Determine which electrodes to use
    if electrodes is not None:
        available_electrodes = [ch for ch in electrodes if ch in epochs.ch_names]
        if len(available_electrodes) == 0:
            print(f"  Warning: None of the requested electrodes {electrodes} found in data.")
            print(f"  Using all available electrodes instead.")
            electrode_subset = None
        else:
            electrode_subset = available_electrodes
            missing = set(electrodes) - set(available_electrodes)
            if missing:
                print(f"  Note: Electrodes not found in data: {missing}")
            print(f"  Using {len(electrode_subset)} electrodes: {electrode_subset}")
    else:
        electrode_subset = None
        print(f"  Using all {len(epochs.ch_names)} electrodes")
    
    # Build title with band info
    electrode_label = f" ({', '.join(electrode_subset)})" if electrode_subset else " (All Channels)"
    band_label = f" - {band_info['label']}" if band_info and band_info['name'] != 'broadband' else ""
    ci_percent = int(ci * 100)
    title = f'{waveform_name} Waveform - All Amplitudes{electrode_label}{band_label} ({ci_percent}% CI)'
    
    # Prepare evoked objects for each amplitude
    evoked_dict = {}
    # colors = {'Base': 'blue', 'Mid (+0.25)': 'orange', 'High (+0.5)': 'red'}
    colors = {'Base': 'blue'}
    
    for trigger_val, amp_label in zip(trigger_values, AMPLITUDE_LABELS):
        try:
            epochs_subset = epochs[epochs.events[:, 2] == trigger_val]
            
            if len(epochs_subset) == 0:
                print(f"  Warning: No epochs found for trigger {trigger_val}")
                continue
            
            # Select electrodes if specified
            if electrode_subset is not None:
                epochs_to_plot = epochs_subset.copy().pick_channels(electrode_subset)
            else:
                epochs_to_plot = epochs_subset
            
            # Compute evoked response
            evoked = epochs_to_plot.average()
            evoked.comment = f'{amp_label} (n={len(epochs_subset)})'
            evoked_dict[amp_label] = evoked
            
        except KeyError:
            print(f"  Warning: Trigger {trigger_val} not found in epochs")
    
    if len(evoked_dict) == 0:
        print(f"  Error: No evoked responses to plot for {waveform_name}")
        return
    
    # Use MNE's plot_compare_evokeds function
    # This function natively supports confidence intervals and professional styling
    fig = mne.viz.plot_compare_evokeds(
        evoked_dict,
        picks='all',  # Average across all selected channels
        combine='mean',  # Show mean across channels
        ci=ci,  # Confidence interval (90%, 95%, etc.)
        show_sensors=True,
        colors=colors,
        title=title,
        show=False,
        split_legend=False,
        truncate_yaxis='auto'
    )
    
    # Enhance the plot
    ax = fig[0].axes[0]
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, 
              label='Stimulus onset')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        fig[0].savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")
    
    # plt.show()
    plt.close(fig[0])


def plot_all_waveforms_comparison(epochs, electrodes=None, band_info=None, save_path=None, ci=0.90):
    """
    Create a comprehensive plot showing all waveforms and amplitudes using MNE
    Shows mean with confidence intervals
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched data
    electrodes : list of str or None
        List of electrode names to include in the plot
    band_info : dict or None
        Dictionary with band information (for plot titles)
    save_path : str, optional
        Path to save the figure
    ci : float, optional
        Confidence interval level (default: 0.90 for 90% CI)
        Options: 0.90 (90%), 0.95 (95%), 0.99 (99%)
    """
    print("\nCreating comprehensive comparison plot...")
    
    # Determine which electrodes to use
    if electrodes is not None:
        available_electrodes = [ch for ch in electrodes if ch in epochs.ch_names]
        if len(available_electrodes) == 0:
            print(f"  Warning: None of the requested electrodes found in data.")
            print(f"  Using all available electrodes instead.")
            electrode_subset = None
        else:
            electrode_subset = available_electrodes
            missing = set(electrodes) - set(available_electrodes)
            if missing:
                print(f"  Note: Electrodes not found: {missing}")
            print(f"  Using {len(electrode_subset)} electrodes: {electrode_subset}")
    else:
        electrode_subset = None
        print(f"  Using all {len(epochs.ch_names)} electrodes")
    
    # Build title with band info
    electrode_label = f" ({', '.join(electrode_subset)})" if electrode_subset else " (All Channels)"
    band_label = f" - {band_info['label']}" if band_info and band_info['name'] != 'broadband' else ""
    ci_percent = int(ci * 100)
    main_title = f'All Waveforms - Evoked Responses Comparison{electrode_label}{band_label} ({ci_percent}% CI)'
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 18))
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.995)
    
    # uncomment if t3
    # colors = {'Base': 'blue', 'Mid (+0.25)': 'orange', 'High (+0.5)': 'red'}
    colors = {'Base': 'blue'}
    # Get z-score for the requested confidence level
    ci_z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z_score = ci_z_scores.get(ci, 1.645)  # Default to 90% if not found
    
    for ax_idx, (waveform_name, trigger_values) in enumerate(WAVEFORM_GROUPS.items()):
        # Prepare evoked objects for this waveform
        evoked_dict = {}
        
        for trigger_val, amp_label in zip(trigger_values, AMPLITUDE_LABELS):
            try:
                epochs_subset = epochs[epochs.events[:, 2] == trigger_val]
                
                if len(epochs_subset) == 0:
                    continue
                
                # Select electrodes if specified
                if electrode_subset is not None:
                    epochs_to_plot = epochs_subset.copy().pick_channels(electrode_subset)
                else:
                    epochs_to_plot = epochs_subset
                
                evoked = epochs_to_plot.average()
                evoked.comment = f'{amp_label} (n={len(epochs_subset)})'
                evoked_dict[amp_label] = evoked
                
            except KeyError:
                continue
        
        if len(evoked_dict) == 0:
            continue
        
        # Plot using MNE's compare evoked on this subplot
        ax = axes[ax_idx]
        
        # Plot each condition
        for amp_label, evoked in evoked_dict.items():
            times = evoked.times * 1000  # Convert to ms
            data = evoked.data
            mean_data = data.mean(axis=0) * 1e6  # Convert to µV
            
            # Calculate confidence intervals (z_score * SEM)
            sem = (data.std(axis=0) / np.sqrt(data.shape[0])) * 1e6
            ci_interval = z_score * sem
            
            color = colors[amp_label]
            
            # Plot mean line
            ax.plot(times, mean_data, 
                   label=evoked.comment,
                   color=color, linewidth=2.5)
            
            # Plot confidence interval as shaded region
            ax.fill_between(times, mean_data - ci_interval, mean_data + ci_interval,
                           color=color, alpha=0.25)
        
        # Enhance subplot
        ax.set_title(f'{waveform_name} Waveform', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_ylabel('Amplitude (µV)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Only show x-label on bottom plot
        if ax_idx == len(WAVEFORM_GROUPS) - 1:
            ax.set_xlabel('Time (ms)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")
    
    # plt.show()
    plt.close(fig)


def plot_amplitude_comparison(epochs, electrodes=None, band_info=None, save_path=None, ci=0.90):
    """
    Create a plot showing all waveforms organized by amplitude level
    
    This function creates 3 panels (one for each amplitude: Base, Mid, High)
    showing all 5 waveforms on each panel. This is the inverse organization
    of plot_all_waveforms_comparison.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Epoched data
    electrodes : list of str or None
        List of electrode names to include in the plot
    band_info : dict or None
        Dictionary with band information (for plot titles)
    save_path : str, optional
        Path to save the figure
    ci : float, optional
        Confidence interval level (default: 0.90 for 90% CI)
        Options: 0.90 (90%), 0.95 (95%), 0.99 (99%)
    
    Returns:
    --------
    None
    
    Example:
    --------
    # Plot all waveforms organized by amplitude level
    plot_amplitude_comparison(
        epochs,
        electrodes=['Cz', 'C1', 'C2'],
        band_info=band_info,
        save_path='./amplitude_comparison.png',
        ci=0.95  # Use 95% confidence intervals
    )
    """
    print("\nCreating amplitude-organized comparison plot...")
    
    # Determine which electrodes to use
    if electrodes is not None:
        available_electrodes = [ch for ch in electrodes if ch in epochs.ch_names]
        if len(available_electrodes) == 0:
            print(f"  Warning: None of the requested electrodes found in data.")
            print(f"  Using all available electrodes instead.")
            electrode_subset = None
        else:
            electrode_subset = available_electrodes
            missing = set(electrodes) - set(available_electrodes)
            if missing:
                print(f"  Note: Electrodes not found: {missing}")
            print(f"  Using {len(electrode_subset)} electrodes: {electrode_subset}")
    else:
        electrode_subset = None
        print(f"  Using all {len(epochs.ch_names)} electrodes")
    
    # Build title with band info
    electrode_label = f" ({', '.join(electrode_subset)})" if electrode_subset else " (All Channels)"
    band_label = f" - {band_info['label']}" if band_info and band_info['name'] != 'broadband' else ""
    ci_percent = int(ci * 100)
    main_title = f'All Waveforms by Amplitude Level{electrode_label}{band_label} ({ci_percent}% CI)'
    
    # Create figure with 3 subplots (one for each amplitude level)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(main_title, fontsize=18, fontweight='bold', y=0.995)
    
    # Colors for each waveform type
    waveform_colors = {
        'Sine': '#1f77b4',      # Blue
        'Square': '#ff7f0e',    # Orange
        'Triangle': '#2ca02c',  # Green
        'LinInc': '#d62728',    # Red
        'LinDec': '#9467bd'     # Purple
    }
    
    # Get z-score for the requested confidence level
    ci_z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z_score = ci_z_scores.get(ci, 1.645)  # Default to 90% if not found
    
    # Organize data by amplitude level
    amplitude_indices = [0, 1, 2]  # Base (0), Mid (+0.25, 1), High (+0.5, 2)
    
    for amp_idx, (amplitude_label, ax) in enumerate(zip(AMPLITUDE_LABELS, axes)):
        ax.set_title(f'{amplitude_label} Amplitude', fontsize=14, fontweight='bold')
        
        # Plot all waveforms for this amplitude level
        for waveform_name, trigger_values in WAVEFORM_GROUPS.items():
            trigger_val = trigger_values[amp_idx]  # Get trigger for this amplitude
            
            try:
                epochs_subset = epochs[epochs.events[:, 2] == trigger_val]
                
                if len(epochs_subset) == 0:
                    continue
                
                # Select electrodes if specified
                if electrode_subset is not None:
                    epochs_to_plot = epochs_subset.copy().pick_channels(electrode_subset)
                else:
                    epochs_to_plot = epochs_subset
                
                evoked = epochs_to_plot.average()
                
                # Get data
                times = evoked.times * 1000  # Convert to ms
                data = evoked.data
                mean_data = data.mean(axis=0) * 1e6  # Convert to µV
                
                # Calculate confidence intervals
                sem = (data.std(axis=0) / np.sqrt(data.shape[0])) * 1e6
                ci_interval = z_score * sem
                
                color = waveform_colors[waveform_name]
                
                # Plot mean line
                ax.plot(times, mean_data, 
                       label=f'{waveform_name} (n={len(epochs_subset)})',
                       color=color, linewidth=2.5)
                
                # Plot confidence interval as shaded region
                ax.fill_between(times, mean_data - ci_interval, mean_data + ci_interval,
                               color=color, alpha=0.2)
                
            except KeyError:
                continue
        
        # Enhance subplot
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.set_ylabel('Amplitude (µV)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Only show x-label on bottom plot
        if amp_idx == len(AMPLITUDE_LABELS) - 1:
            ax.set_xlabel('Time (ms)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")
    
    # plt.show()
    plt.close(fig)


def plot_topographic_maps(epochs, waveform_name, trigger_values, times='auto', save_path=None):
    """
    Plot topographic maps for a waveform at specific time points
    """
    print(f"\nCreating topographic maps for {waveform_name}...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'{waveform_name} - Topographic Maps', 
                 fontsize=16, fontweight='bold')
    
    for idx, (trigger_val, amp_label) in enumerate(zip(trigger_values, AMPLITUDE_LABELS)):
        try:
            epochs_subset = epochs[epochs.events[:, 2] == trigger_val]
            
            if len(epochs_subset) == 0:
                continue
            
            evoked = epochs_subset.average()
            
            # Plot topographic map
            evoked.plot_topomap(times=times, axes=axes[idx:idx+1], 
                               show=False, colorbar=True,
                               time_unit='s')
            axes[idx].set_title(f'{amp_label} (n={len(epochs_subset)})', 
                               fontsize=12)
            
        except KeyError:
            continue
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {save_path}")
    
    # plt.show()


# ==============================================================================
# MAIN ANALYSIS FUNCTION (UPDATED WITH SAVE/LOAD)
# ==============================================================================

def analyze_eeg_waveforms(xdf_filepath, output_dir='./output', 
                         apply_ica=True, n_ica_components=15, plot_ica=True,
                         electrode_groups=None, frequency_band='broadband',
                         save_preprocessed=False, subject_id=None, session_id=None):
    """
    Main function to analyze EEG waveform data
    
    Parameters:
    -----------
    xdf_filepath : str
        Path to the XDF file
    output_dir : str
        Directory to save output figures
    apply_ica : bool
        Whether to apply ICA for artifact removal (default: True)
    n_ica_components : int
        Number of ICA components to compute (default: 15)
    plot_ica : bool
        Whether to generate ICA component plots (default: True)
    electrode_groups : dict, list, str, or None
        Electrodes to plot
    frequency_band : str
        Frequency band to extract (default: 'broadband')
    save_preprocessed : bool
        Save preprocessed data for later use (NEW!)
    subject_id : str or int
        Subject ID (required if save_preprocessed=True)
    session_id : str
        Session ID (required if save_preprocessed=True)
    
    Returns:
    --------
    raw_preprocessed : mne.io.Raw
        Preprocessed raw data
    epochs : mne.Epochs
        Epoched data
    ica : mne.preprocessing.ICA or None
        ICA solution
    """
    import shutil
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("EEG WAVEFORM ANALYSIS PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    raw, events = load_xdf_to_mne(xdf_filepath)
    
    # Step 2: Preprocess (including ICA)
    raw_preprocessed, ica = preprocess_eeg(raw, apply_ica=apply_ica, 
                                          n_components=n_ica_components,
                                          plot_ica=plot_ica)
    
    # Save ICA plots to output directory if ICA was applied
    if ica is not None and plot_ica:
        for fname in ['ica_components_topomap.png', 'ica_sources_timeseries.png', 
                     'ica_excluded_components.png', 'ica_components_report.txt']:
            if os.path.exists(fname):
                shutil.move(fname, os.path.join(output_dir, fname))
    
    # Step 3: Apply frequency band filtering
    raw_band, band_info = apply_frequency_band(raw_preprocessed, frequency_band)
    
    # Step 4: Create epochs
    epochs = create_epochs(raw_band, events, TRIGGER_MAPPING)
    
    # NEW: Save preprocessed data if requested
    if save_preprocessed:
        if subject_id is None or session_id is None:
            print("\nWarning: Cannot save preprocessed data without subject_id and session_id")
        else:
            save_preprocessed_data(raw_preprocessed, epochs, ica,
                                  output_dir, subject_id, session_id)
    
    # Step 5: Process electrode groups
    if electrode_groups is None:
        plot_groups = {'all': None}
    elif isinstance(electrode_groups, str):
        if electrode_groups in ELECTRODE_GROUPS:
            plot_groups = {electrode_groups: ELECTRODE_GROUPS[electrode_groups]}
        else:
            print(f"Warning: Unknown electrode group '{electrode_groups}'")
            print(f"Available groups: {list(ELECTRODE_GROUPS.keys())}")
            print("Using all electrodes instead.")
            plot_groups = {'all': None}
    elif isinstance(electrode_groups, list):
        plot_groups = {'custom': electrode_groups}
    elif isinstance(electrode_groups, dict):
        plot_groups = electrode_groups
    else:
        print("Warning: Invalid electrode_groups format. Using all electrodes.")
        plot_groups = {'all': None}
    
    # Step 6: Plot individual waveforms for each electrode group
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    
    for group_name, electrodes in plot_groups.items():
        print(f"\nPlotting electrode group: {group_name}")
        if band_info['name'] != 'broadband':
            print(f"Frequency band: {band_info['label']}")
        print("-" * 60)
        
        # Create subdirectory for this electrode group if multiple groups
        if len(plot_groups) > 1:
            group_output_dir = os.path.join(output_dir, f'group_{group_name}')
            os.makedirs(group_output_dir, exist_ok=True)
        else:
            group_output_dir = output_dir
        
        # Create filename suffix with electrode and band info
        electrode_suffix = f"_{group_name}" if electrodes is not None and group_name != 'all' else ""
        band_suffix = f"_{band_info['name']}_{band_info['range'][0]}-{band_info['range'][1]}Hz" if band_info['name'] != 'broadband' else ""
        
        # Plot individual waveforms
        for waveform_name, trigger_values in WAVEFORM_GROUPS.items():
            filename = f'{waveform_name}_evoked{electrode_suffix}{band_suffix}.png'
            save_path = os.path.join(group_output_dir, filename)
            plot_waveform_evoked(epochs, waveform_name, trigger_values, 
                               electrodes=electrodes, band_info=band_info,
                               save_path=save_path)
        
        # Create comparison plot
        filename = f'all_waveforms_comparison{electrode_suffix}{band_suffix}.png'
        save_path = os.path.join(group_output_dir, filename)
        plot_all_waveforms_comparison(epochs, electrodes=electrodes,
                                     band_info=band_info, save_path=save_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return raw_preprocessed, epochs, ica


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    # Specify your XDF file path
    Subject_id = 3
    while Subject_id <= 12:
    
        XDF_FILE = f"EEG_Practice/Subject{Subject_id}_data/EEGDATA_SUB{Subject_id}-T4.xdf"
        OUTPUT_DIR = f"./eeg_analysis_output_Subject{Subject_id}_Task4"

        # ========================================================================
        # EXAMPLE 1: Basic analysis with SAVE
        # ========================================================================
        raw, epochs, ica = analyze_eeg_waveforms(
            XDF_FILE,
            OUTPUT_DIR,
            apply_ica=True,
            n_ica_components=15,
            electrode_groups='central',
            frequency_band='alpha',
            save_preprocessed=True,  # ← NEW: Save for reuse!
            subject_id=Subject_id,
            session_id='Task4'
        )
        Subject_id += 1
    
    # ========================================================================
    # EXAMPLE 2: LOAD preprocessed data (fast!)
    # ========================================================================
    # raw, epochs, ica = load_preprocessed_data(
    #     OUTPUT_DIR, Subject_id, 'Task3'
    # )
    # 
    # # Now analyze different frequency band (takes ~10 seconds!)
    # raw_theta, band_info = apply_frequency_band(raw, 'theta')
    # epochs_theta = create_epochs(raw_theta, epochs.events, TRIGGER_MAPPING)
    # # Plot...
    
    # ========================================================================
    # EXAMPLE 3: Analyze multiple bands using saved data
    # ========================================================================
    # for band in ['theta', 'alpha', 'beta']:
    #     raw, epochs, ica = load_preprocessed_data(OUTPUT_DIR, Subject_id, 'Task3')
    #     raw_band, band_info = apply_frequency_band(raw, band)
    #     epochs_band = create_epochs(raw_band, epochs.events, TRIGGER_MAPPING)
    #     # Plot for each band...