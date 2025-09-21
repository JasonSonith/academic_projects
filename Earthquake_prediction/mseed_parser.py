from obspy import read, UTCDateTime
import numpy as np
import pandas as pd
from pathlib import Path
import os

def process_mseed_data(file_path, start_time=None, end_time=None, freqmin=1.0, freqmax=20.0, output_dir="processed_data"):
    """
    Complete MSEED data processing pipeline:
    - Read MSEED data
    - Select Z channel
    - Merge gaps/overlaps
    - Trim data
    - Detrend
    - Bandpass filter
    - Cast to float
    - Save numpy array and CSV
    """
    
    print(f"Processing file: {file_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Read MSEED data
    st = read(file_path)
    print(f"Original traces: {len(st)}")
    
    # 2. Select Z channel (vertical component)
    st = st.select(channel="*Z")
    print(f"After Z channel selection: {len(st)} traces")
    
    # 3. Merge gaps and overlaps
    st.merge(method=1, fill_value="interpolate")
    print(f"After merging: {len(st)} traces") #should be 1 trace
    
    # 4. Trim data (if time range specified)
    if start_time and end_time:
        st.trim(starttime=UTCDateTime(start_time), endtime=UTCDateTime(end_time))
        print(f"Trimmed to: {start_time} - {end_time}")
    
    # 5. Detrend (remove linear trend)
    st.detrend(type='linear') #this removes linear trend which is noise in the data
    print("Applied linear detrending")
    
    # 6. Bandpass filter which removes low and high frequency noise
    st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    print(f"Applied bandpass filter: {freqmin}-{freqmax} Hz")
    
    # 7. Cast to float32 which is the data type of the data
    for trace in st:
        trace.data = trace.data.astype(np.float32)
    print("Cast to float32")
    
    # 8. Get processed data which is the data after the processing
    if len(st) > 0:
        trace = st[0]
        data = trace.data
        times = np.arange(len(data)) / trace.stats.sampling_rate
        
        # 9. Save numpy array which is the data after the processing      
        np_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_processed.npy")
        np.save(np_filename, data)
        print(f"Saved numpy array: {np_filename}")
        
        # 10. Save CSV with time and data
        csv_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_processed.csv")
        df = pd.DataFrame({
            'time_seconds': times,
            'amplitude': data,
            'timestamp': [trace.stats.starttime + i/trace.stats.sampling_rate for i in range(len(data))]
        })
        df.to_csv(csv_filename, index=False)
        print(f"Saved CSV: {csv_filename}")
        
        # Print summary statistics
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

# Process all MSEED files
if __name__ == "__main__":
    # Find all mseed files in current directory
    mseed_files = [f for f in os.listdir('.') if f.endswith('.mseed')]
    print(f"Found {len(mseed_files)} mseed files to process: {mseed_files}")
    
    # Process each file
    for mseed_file in mseed_files:
        print(f"\n{'='*60}")
        print(f"Processing: {mseed_file}")
        print(f"{'='*60}")
        
        try:
            st, data, df = process_mseed_data(mseed_file)
            if st is not None:
                print(f"✓ Successfully processed {mseed_file}")
            else:
                print(f"✗ Failed to process {mseed_file}")
        except Exception as e:
            print(f"✗ Error processing {mseed_file}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")