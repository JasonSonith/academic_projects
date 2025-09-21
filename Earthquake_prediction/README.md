# Earthquake Data Processing Program Overview

## What It Does (Step by Step):

### 1. Takes Raw Earthquake Data
- Reads the `.mseed` files (raw earthquake sensor recordings)
- Like taking a messy recording and cleaning it up

### 2. Cleans the Data
- Removes noise and unwanted signals
- Keeps only the important earthquake frequencies (1-20 Hz)
- Fixes any gaps or problems in the data

### 3. Detects Earthquakes
- Looks through the data and says "This part is an earthquake" or "This part is just normal noise"
- Like having a smart assistant that spots earthquakes automatically

### 4. Adds Labels and Information
- Adds tags to each data point: "earthquake detected: YES/NO"
- Adds station info: "Recorded at Station X in California"
- Adds event info: "This was a magnitude 3.2 earthquake"

### 5. Creates Multiple Output Files
- **Individual CSV files**: One clean file per earthquake recording
- **Individual NPY files**: Fast-loading versions for analysis
- **Master dataset**: One big file combining everything

## What You Get:

### Before: Raw, messy earthquake data files
### After: Clean, labeled, organized dataset ready for:
- Machine learning
- Earthquake analysis
- Research
- Pattern detection

## Output Files Created:

### Individual Files (per mseed file):
- `filename_processed.csv` - Human-readable data with earthquake labels
- `filename_processed.npy` - Fast-loading binary data

### Master Dataset Files:
- `master_dataset.csv` - All data combined in one file
- `master_dataset_amplitudes.npy` - All amplitude data in binary format

### What's in Each CSV File:
- `time_seconds` - Time from start of recording
- `amplitude` - Earthquake signal strength
- `timestamp` - Actual date/time of measurement
- `earthquake_detected` - True/False for each data point
- `event_magnitude` - Estimated magnitude of events
- `station_id` - Which station recorded this
- `channel` - Direction recorded (Z/N/E)
- `location` - Station location
- `latitude/longitude` - Station coordinates
- `event_type` - Type of seismic event
- `magnitude` - Known event magnitude
- `depth` - Event depth in km
