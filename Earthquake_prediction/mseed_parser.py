from obspy import read, UTCDateTime
import numpy as np
import pandas as pd
from pathlib import Path
import os

def detect_earthquake_events(data, sampling_rate, threshold_multiplier=3.0):
    """Simple earthquake detection based on signal amplitude."""
    noise_level = np.std(data)
    threshold = threshold_multiplier * noise_level
    earthquake_detected = np.abs(data) > threshold
    event_magnitude = np.where(earthquake_detected, np.log10(np.abs(data) + 1) * 2, 0)
    return earthquake_detected, event_magnitude

def create_master_dataset(processed_files, output_dir="processed_data"):
    """Creates a master dataset file combining all processed earthquake data."""
    print(f"\nCreating master dataset from {len(processed_files)} files...")
    
    all_dataframes = []
    for csv_file in processed_files:
        if os.path.exists(csv_file):
            print(f"Adding {os.path.basename(csv_file)} to master dataset...")
            df = pd.read_csv(csv_file)
            df['source_file'] = os.path.basename(csv_file)
            all_dataframes.append(df)
        else:
            print(f"Warning: {csv_file} not found, skipping...")
    
    if not all_dataframes:
        print("No processed files found to create master dataset!")
        return None
    
    master_df = pd.concat(all_dataframes, ignore_index=True)
    master_df = master_df.sort_values('timestamp').reset_index(drop=True)
    
    master_csv_path = os.path.join(output_dir, "master_dataset.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f" Master dataset saved as CSV: {master_csv_path}")
    
    amplitude_data = master_df['amplitude'].values
    master_npy_path = os.path.join(output_dir, "master_dataset_amplitudes.npy")
    np.save(master_npy_path, amplitude_data)
    print(f" Master dataset amplitudes saved as NPY: {master_npy_path}")
    
    print(f"\nMaster Dataset Summary:")
    print(f"Total data points: {len(master_df):,}")
    print(f"Time range: {master_df['timestamp'].min()} to {master_df['timestamp'].max()}")
    print(f"Earthquake events detected: {master_df['earthquake_detected'].sum():,}")
    print(f"Unique stations: {master_df['station_id'].nunique()}")
    print(f"Unique channels: {master_df['channel'].unique()}")
    
    return master_df

def process_mseed_data(file_path, start_time=None, end_time=None, freqmin=1.0, freqmax=20.0, channel="*Z", output_dir="processed_data", station_info=None, event_info=None):
    """Complete MSEED data processing pipeline for earthquake data."""
    print(f"Processing file: {file_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    st = read(file_path)
    print(f"Original traces: {len(st)}")
    
    st = st.select(channel=channel)
    print(f"After {channel} channel selection: {len(st)} traces")
    
    st.merge(method=1, fill_value="interpolate")
    print(f"After merging: {len(st)} traces")
    
    if start_time and end_time:
        st.trim(starttime=UTCDateTime(start_time), endtime=UTCDateTime(end_time))
        print(f"Trimmed to: {start_time} - {end_time}")
    
    st.detrend(type='linear')
    print("Applied linear detrending")
    
    st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    print(f"Applied bandpass filter: {freqmin}-{freqmax} Hz")
    
    for trace in st:
        trace.data = trace.data.astype(np.float32)
    print("Cast to float32")
    
    if len(st) > 0:
        trace = st[0]
        data = trace.data
        times = np.arange(len(data)) / trace.stats.sampling_rate
        
        np_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_processed.npy")
        np.save(np_filename, data)
        print(f"Saved numpy array: {np_filename}")
        
        earthquake_detected, event_magnitude = detect_earthquake_events(data, trace.stats.sampling_rate)
        
        csv_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_processed.csv")
        
        if station_info is None:
            station_info = {
                'station_id': trace.stats.station if hasattr(trace.stats, 'station') else 'UNKNOWN',
                'location': 'Unknown Location',
                'latitude': 0.0,
                'longitude': 0.0
            }
        
        if event_info is None:
            event_info = {
                'event_type': 'unknown',
                'magnitude': 0.0,
                'depth': 0.0
            }
        
        df = pd.DataFrame({
            'time_seconds': times,
            'amplitude': data,
            'timestamp': [trace.stats.starttime + i/trace.stats.sampling_rate for i in range(len(data))],
            'earthquake_detected': earthquake_detected,
            'event_magnitude': event_magnitude,
            'station_id': station_info['station_id'],
            'channel': trace.stats.channel,
            'location': station_info['location'],
            'latitude': station_info['latitude'],
            'longitude': station_info['longitude'],
            'event_type': event_info['event_type'],
            'magnitude': event_info['magnitude'],
            'depth': event_info['depth']
        })
        
        df.to_csv(csv_filename, index=False)
        print(f"Saved enhanced CSV with earthquake detection: {csv_filename}")
        
        print(f"\nData Summary:")
        print(f"Station: {trace.stats.station}")
        print(f"Channel: {trace.stats.channel}")
        print(f"Start time: {trace.stats.starttime}")
        print(f"End time: {trace.stats.endtime}")
        print(f"Duration: {trace.stats.endtime - trace.stats.starttime} seconds")
        print(f"Sample rate: {trace.stats.sampling_rate} Hz")
        print(f"Number of samples: {len(data)}")
        print(f"Data range: {np.min(data):.6f} to {np.max(data):.6f}")
        print(f"Data mean: {np.mean(data):.6f}")
        print(f"Data std: {np.std(data):.6f}")
        
        return st, data, df
    else:
        print("No data found after processing!")
        return None, None, None

if __name__ == "__main__":
    mseed_dir = "mseed_data"
    channel = "*Z"
    freqmin = 1.0
    freqmax = 20.0
    
    if not os.path.exists(mseed_dir):
        print(f"Error: {mseed_dir} directory not found!")
        exit(1)
    
    mseed_files = [f for f in os.listdir(mseed_dir) if f.endswith('.mseed')]
    print(f"Found {len(mseed_files)} mseed files to process: {mseed_files}")
    print(f"Processing channel: {channel}")
    print(f"Bandpass filter: {freqmin}-{freqmax} Hz")
    
    processed_csv_files = []
    
    for mseed_file in mseed_files:
        file_path = os.path.join(mseed_dir, mseed_file)
        print(f"\n{'='*60}")
        print(f"Processing: {mseed_file}")
        print(f"{'='*60}")
        
        try:
            station_info = {
                'station_id': 'STATION_001',
                'location': 'California, USA',
                'latitude': 37.7749,
                'longitude': -122.4194
            }
            
            event_info = {
                'event_type': 'earthquake',
                'magnitude': 3.2,
                'depth': 10.5
            }
            
            st, data, df = process_mseed_data(file_path, channel=channel, freqmin=freqmin, freqmax=freqmax, station_info=station_info, event_info=event_info)
            
            if st is not None:
                print(f"Successfully processed {mseed_file}")
                csv_filename = os.path.join("processed_data", f"{os.path.splitext(mseed_file)[0]}_processed.csv")
                processed_csv_files.append(csv_filename)
            else:
                print(f" Failed to process {mseed_file}")
        except Exception as e:
            print(f" Error processing {mseed_file}: {str(e)}")
    
    if processed_csv_files:
        print(f"\n{'='*60}")
        print("CREATING MASTER DATASET")
        print(f"{'='*60}")
        
        master_df = create_master_dataset(processed_csv_files)
        
        if master_df is not None:
            print(f"\n Master dataset created successfully!")
            print(f"  - Individual files: {len(processed_csv_files)}")
            print(f"  - Total data points: {len(master_df):,}")
            print(f"  - Earthquake events: {master_df['earthquake_detected'].sum():,}")
        else:
            print(" Failed to create master dataset")
    
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print("Your enhanced earthquake dataset includes:")
    print(" Individual CSV files with earthquake detection labels")
    print(" Individual NPY files for fast data loading")
    print(" Master dataset CSV combining all files")
    print(" Master dataset NPY with amplitude data")
    print("Station and event metadata")
    print(" Earthquake event detection and magnitude estimates")
    print("\nPerfect for machine learning and earthquake analysis!")
    print(f"{'='*60}")