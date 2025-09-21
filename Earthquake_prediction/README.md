# Earthquake Data Parsing – README

## What the Program Does
The `mseed_parser.py` program takes raw earthquake sensor data (`.mseed` files) and cleans them up for analysis by removing noise. T

### The Cleaning Process
1. **Reads** the raw earthquake data.  
2. **Filters** to only use the vertical (Z) channel (most important for earthquakes because it measures the up and down motion).  
3. **Removes** gaps and overlaps in the data.  
4. **Detrends** – removes slow drift/noise.  
5. **Bandpass filters** – keeps only frequencies between **1–20 Hz** (removes very low and high frequency noise).  
6. **Converts** data to the right format (**float32**).  

---

## CSV Files Output
The CSV files contain your cleaned earthquake data in a spreadsheet format with **3 columns**:

- **time_seconds**: Time from the start of recording (0.0, 0.01, 0.02, etc.).  
- **amplitude**: The earthquake signal strength at each time point.  
- **timestamp**: The actual date/time when each measurement was taken.  

> Each file has ~**90,000 data points** (about **15 minutes** of data at **100 samples per second**).

---

## NPY Files
The `.npy` files are **NumPy arrays** — just the raw amplitude data **without** the time information.

Think of them as:
- **Super fast** to load in Python.  
- **Memory efficient** for calculations.  
- **Perfect for machine learning** — just the numbers you need for analysis.  

### Why Both Formats?
- **CSV**: Human‑readable, easy to plot, great for quick analysis.  
- **NPY**: Blazing fast loading, ideal for feeding into ML models or doing heavy computations.  

It’s like having both a detailed report (**CSV**) and a quick reference card (**NPY**) of the same information!
