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
    event_magnitude = np.where(earthquake_detected, 
                              np.log10(np.abs(data) + 1) * 2,
                              0)
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
        
        # STEP 11: Print summary statistics about the processed data
        # This helps you understand what you're working with
        print(f"\nData Summary:")
        print(f"Station: {trace.stats.station}")           # Which earthquake station recorded this
        print(f"Channel: {trace.stats.channel}")             # Which direction (Z/N/E) was recorded
        print(f"Start time: {trace.stats.starttime}")        # When the recording started
        print(f"End time: {trace.stats.endtime}")            # When the recording ended
        print(f"Duration: {trace.stats.endtime - trace.stats.starttime} seconds")  # Total recording time
        print(f"Sample rate: {trace.stats.sampling_rate} Hz")  # How many measurements per second
        print(f"Number of samples: {len(data)}")             # Total number of data points
        print(f"Data range: {np.min(data):.6f} to {np.max(data):.6f}")  # Smallest and largest values
        print(f"Data mean: {np.mean(data):.6f}")             # Average value (should be close to 0 after detrending)
        print(f"Data std: {np.std(data):.6f}")               # Standard deviation (measure of signal strength)
        
        # Return the processed data for further analysis
        return st, data, df
    else:
        print("No data found after processing!")
        return None, None, None

# =============================================================================
# MAIN PROGRAM: Process all MSEED files in your dataset
# =============================================================================
# This section runs when you execute the script directly (not when importing it)
if __name__ == "__main__":
    
    # =========================================================================
    # CONFIGURATION SETTINGS - Change these to customize your processing
    # =========================================================================
    mseed_dir = "mseed_data"        # Folder containing your raw MSEED files
    channel = "*Z"                  # Which channel to process:
                                   #   "*Z" = Vertical (up/down motion) - most common for earthquakes
                                   #   "*N" = North-South horizontal motion
                                   #   "*E" = East-West horizontal motion
                                   #   "*"  = All channels (processes everything)
    freqmin = 1.0                  # Minimum frequency for filtering (Hz)
    freqmax = 20.0                 # Maximum frequency for filtering (Hz)
    
    # =========================================================================
    # FIND ALL MSEED FILES TO PROCESS
    # =========================================================================
    # Check if the mseed_data folder exists
    if not os.path.exists(mseed_dir):
        print(f"Error: {mseed_dir} directory not found!")
        print("Make sure you have a folder called 'mseed_data' with your MSEED files inside.")
        exit(1)
    
    # Find all files ending with .mseed in the mseed_data folder
    mseed_files = [f for f in os.listdir(mseed_dir) if f.endswith('.mseed')]
    print(f"Found {len(mseed_files)} mseed files to process: {mseed_files}")
    print(f"Processing channel: {channel}")
    print(f"Bandpass filter: {freqmin}-{freqmax} Hz")
    
    # =========================================================================
    # PROCESS EACH MSEED FILE WITH ENHANCED FEATURES
    # =========================================================================
    # List to store processed CSV files for master dataset creation
    processed_csv_files = []
    
    # Loop through each MSEED file and process it
    for mseed_file in mseed_files:
        file_path = os.path.join(mseed_dir, mseed_file)
        print(f"\n{'='*60}")
        print(f"Processing: {mseed_file}")
        print(f"{'='*60}")
        
        # Try to process the file, catch any errors that might occur
        try:
            # Example station and event metadata (you can customize these)
            station_info = {
                'station_id': 'STATION_001',  # You can extract this from filename or metadata
                'location': 'California, USA',
                'latitude': 37.7749,
                'longitude': -122.4194
            }
            
            event_info = {
                'event_type': 'earthquake',  # Could be 'earthquake', 'noise', 'explosion', etc.
                'magnitude': 3.2,           # Known magnitude if available
                'depth': 10.5               # Event depth in km
            }
            
            # Call our enhanced processing function with earthquake detection and metadata
            st, data, df = process_mseed_data(
                file_path, 
                channel=channel, 
                freqmin=freqmin, 
                freqmax=freqmax,
                station_info=station_info,
                event_info=event_info
            )
            
            # Check if processing was successful
            if st is not None:
                print(f"Successfully processed {mseed_file}")
                
                # Add the processed CSV file to our list for master dataset
                csv_filename = os.path.join("processed_data", f"{os.path.splitext(mseed_file)[0]}_processed.csv")
                processed_csv_files.append(csv_filename)
            else:
                print(f" Failed to process {mseed_file}")
        except Exception as e:
            # If something goes wrong, print the error but continue with other files
            print(f" Error processing {mseed_file}: {str(e)}")
    
    # =========================================================================
    # CREATE MASTER DATASET
    # =========================================================================
    # After processing all files, create a master dataset combining everything
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
    
    # =========================================================================
    # COMPLETION MESSAGE
    # =========================================================================
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