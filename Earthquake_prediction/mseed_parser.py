# Import required libraries for earthquake data processing
from obspy import read, UTCDateTime  # obspy: library for reading seismic data files
import numpy as np                   # numpy: for numerical operations and arrays
import pandas as pd                  # pandas: for data manipulation and CSV files
from pathlib import Path            # pathlib: for handling file paths
import os                           # os: for operating system functions like creating directories

def detect_earthquake_events(data, sampling_rate, threshold_multiplier=3.0):
    """
    Simple earthquake detection based on signal amplitude.
    
    This function identifies potential earthquake events by looking for:
    - Sudden increases in signal amplitude
    - Sustained high-amplitude periods
    
    Parameters:
    - data: The processed earthquake signal data
    - sampling_rate: How many samples per second
    - threshold_multiplier: How many times above normal noise to trigger detection
    
    Returns:
    - earthquake_detected: Boolean array indicating earthquake events
    - event_magnitude: Estimated magnitude of each event
    """
    
    # Calculate the standard deviation (measure of normal noise level)
    noise_level = np.std(data)
    
    # Set threshold for earthquake detection
    # If signal is more than threshold_multiplier times the noise level, it's likely an earthquake
    threshold = threshold_multiplier * noise_level
    
    # Create boolean array: True where amplitude exceeds threshold
    earthquake_detected = np.abs(data) > threshold
    
    # Calculate estimated magnitude for each point (simplified)
    # This is a rough estimate based on signal amplitude
    event_magnitude = np.where(earthquake_detected, 
                              np.log10(np.abs(data) + 1) * 2,  # Rough magnitude estimate
                              0)  # No earthquake = magnitude 0
    
    return earthquake_detected, event_magnitude

def create_master_dataset(processed_files, output_dir="processed_data"):
    """
    Creates a master dataset file combining all processed earthquake data.
    
    This function:
    - Combines all individual CSV files into one master file
    - Adds a source_file column to track which file each row came from
    - Creates both CSV and NPY versions of the master dataset
    - Perfect for machine learning and analysis
    
    Parameters:
    - processed_files: List of processed CSV file paths
    - output_dir: Directory where master dataset will be saved
    """
    
    print(f"\nCreating master dataset from {len(processed_files)} files...")
    
    # List to store all DataFrames
    all_dataframes = []
    
    # Read each processed CSV file and combine them
    for csv_file in processed_files:
        if os.path.exists(csv_file):
            print(f"Adding {os.path.basename(csv_file)} to master dataset...")
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Add source file information
            df['source_file'] = os.path.basename(csv_file)
            
            # Add to our list
            all_dataframes.append(df)
        else:
            print(f"Warning: {csv_file} not found, skipping...")
    
    if not all_dataframes:
        print("No processed files found to create master dataset!")
        return None
    
    # Combine all DataFrames into one master dataset
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort by timestamp to keep data in chronological order
    master_df = master_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save master dataset as CSV
    master_csv_path = os.path.join(output_dir, "master_dataset.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"✓ Master dataset saved as CSV: {master_csv_path}")
    
    # Create NPY version with just the amplitude data for fast loading
    amplitude_data = master_df['amplitude'].values
    master_npy_path = os.path.join(output_dir, "master_dataset_amplitudes.npy")
    np.save(master_npy_path, amplitude_data)
    print(f"✓ Master dataset amplitudes saved as NPY: {master_npy_path}")
    
    # Print master dataset summary
    print(f"\nMaster Dataset Summary:")
    print(f"Total data points: {len(master_df):,}")
    print(f"Time range: {master_df['timestamp'].min()} to {master_df['timestamp'].max()}")
    print(f"Earthquake events detected: {master_df['earthquake_detected'].sum():,}")
    print(f"Unique stations: {master_df['station_id'].nunique()}")
    print(f"Unique channels: {master_df['channel'].unique()}")
    
    return master_df

def process_mseed_data(file_path, start_time=None, end_time=None, freqmin=1.0, freqmax=20.0, channel="*Z", output_dir="processed_data", station_info=None, event_info=None):
    """
    Complete MSEED data processing pipeline for earthquake data:
    
    What this function does:
    - Reads raw earthquake sensor data from MSEED files
    - Selects a specific channel (Z=vertical, N=north, E=east)
    - Merges gaps and overlaps in the data
    - Trims data to specific time range (optional)
    - Removes linear trends (detrending)
    - Applies bandpass filter to remove noise
    - Converts data to proper format
    - Saves processed data as both CSV and NPY files
    
    Parameters:
    - file_path: Path to the MSEED file to process
    - start_time: Optional start time for data trimming (format: 'YYYY-MM-DDTHH:MM:SS')
    - end_time: Optional end time for data trimming
    - freqmin: Minimum frequency for bandpass filter (Hz)
    - freqmax: Maximum frequency for bandpass filter (Hz)
    - channel: Which channel to process ('*Z'=vertical, '*N'=north, '*E'=east)
    - output_dir: Directory to save processed files
    - station_info: Dictionary with station metadata (station_id, location, etc.)
    - event_info: Dictionary with event metadata (event_type, magnitude, etc.)
    """
    
    print(f"Processing file: {file_path}")
    
    # Create output directory if it doesn't exist
    # exist_ok=True means don't error if directory already exists
    os.makedirs(output_dir, exist_ok=True)
    
    # STEP 1: Read the raw MSEED data file
    # MSEED files contain earthquake sensor data with multiple channels and time series
    st = read(file_path)
    print(f"Original traces: {len(st)}")  # Shows how many data streams are in the file
    
    # STEP 2: Select the specific channel we want to analyze
    # Earthquake sensors typically have 3 channels: Z (vertical), N (north), E (east)
    # We choose one channel because each shows different aspects of ground motion
    st = st.select(channel=channel)
    print(f"After {channel} channel selection: {len(st)} traces")
    
    # STEP 3: Merge any gaps or overlaps in the data
    # Sometimes data has small gaps or overlapping sections that need to be fixed
    # method=1 means use interpolation to fill gaps
    st.merge(method=1, fill_value="interpolate")
    print(f"After merging: {len(st)} traces")  # Should be 1 trace after merging
    
    # STEP 4: Trim data to specific time range (optional)
    # This step is only used if you want to analyze a specific time period
    # For example, you might want to focus on a 5-minute window around an earthquake
    if start_time and end_time:
        st.trim(starttime=UTCDateTime(start_time), endtime=UTCDateTime(end_time))
        print(f"Trimmed to: {start_time} - {end_time}")
    
    # STEP 5: Remove linear trends (detrending)
    # Raw earthquake data often has slow, gradual changes that aren't real earthquakes
    # This step removes these trends to focus on actual seismic signals
    st.detrend(type='linear')  # Removes linear trends which are usually noise
    print("Applied linear detrending")
    
    # STEP 6: Apply bandpass filter to remove unwanted frequencies
    # Earthquake signals are typically between 1-20 Hz, so we filter out:
    # - Very low frequencies (below 1 Hz) = slow ground movements, not earthquakes
    # - Very high frequencies (above 20 Hz) = electronic noise, not earthquakes
    st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    print(f"Applied bandpass filter: {freqmin}-{freqmax} Hz")
    
    # STEP 7: Convert data to float32 format
    # This ensures consistent data type for calculations and saves memory
    # float32 is precise enough for earthquake analysis and uses less memory than float64
    for trace in st:
        trace.data = trace.data.astype(np.float32)
    print("Cast to float32")
    
    # STEP 8: Extract the processed data for saving
    # Check if we have data after all the processing steps
    if len(st) > 0:
        # Get the first (and should be only) trace after processing
        trace = st[0]
        data = trace.data  # This is our cleaned earthquake signal
        
        # Create time array: each data point corresponds to a specific time
        # sampling_rate tells us how many measurements per second (e.g., 100 Hz = 100 measurements/second)
        times = np.arange(len(data)) / trace.stats.sampling_rate
        
        # STEP 9: Save data as NumPy array (.npy file)
        # This format is fast to load and perfect for machine learning
        # The filename includes the original filename + "_processed"
        np_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_processed.npy")
        np.save(np_filename, data)
        print(f"Saved numpy array: {np_filename}")
        
        # STEP 10: Detect earthquake events in the data
        # This adds earthquake detection to help with machine learning
        earthquake_detected, event_magnitude = detect_earthquake_events(data, trace.stats.sampling_rate)
        
        # STEP 11: Save enhanced data as CSV file with earthquake labels and metadata
        # CSV files are human-readable and can be opened in Excel or other programs
        csv_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_processed.csv")
        
        # Create a comprehensive DataFrame with earthquake detection and metadata:
        # - time_seconds: Time from start of recording (0, 0.01, 0.02, etc.)
        # - amplitude: The earthquake signal strength at each time point
        # - timestamp: The actual date/time when each measurement was taken
        # - earthquake_detected: True/False for each data point (NEW!)
        # - event_magnitude: Estimated magnitude of earthquake events (NEW!)
        # - station_id: Which station recorded this data (NEW!)
        # - channel: Which direction was recorded (NEW!)
        # - location: Where the station is located (NEW!)
        # - event_type: Type of seismic event (NEW!)
        
        # Set default metadata if not provided
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
        
        # Create the enhanced DataFrame with all the new columns
        df = pd.DataFrame({
            'time_seconds': times,                    # Time in seconds from start
            'amplitude': data,                        # The earthquake signal values
            'timestamp': [trace.stats.starttime + i/trace.stats.sampling_rate for i in range(len(data))],
            'earthquake_detected': earthquake_detected,  # NEW: True/False for each point
            'event_magnitude': event_magnitude,          # NEW: Estimated magnitude
            'station_id': station_info['station_id'],   # NEW: Station identifier
            'channel': trace.stats.channel,             # NEW: Channel (Z/N/E)
            'location': station_info['location'],        # NEW: Station location
            'latitude': station_info['latitude'],       # NEW: Station latitude
            'longitude': station_info['longitude'],      # NEW: Station longitude
            'event_type': event_info['event_type'],      # NEW: Type of event
            'magnitude': event_info['magnitude'],        # NEW: Event magnitude
            'depth': event_info['depth']                 # NEW: Event depth
        })
        
        df.to_csv(csv_filename, index=False)  # Save without row numbers
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