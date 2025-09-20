from obspy import read, UTCDateTime
import numpy as np

st = read('10-45-00.003.mseed')
st = st.select(channel="*Z") # select the Z channel which is the vertical component (most sensitive to ground motion)
print(st)