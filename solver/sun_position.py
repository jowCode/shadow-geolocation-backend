import pvlib
import pandas as pd
from datetime import datetime

def calculate_sun_position(lat, lon, date, time_utc):
    """
    Berechnet Sonnenposition (Azimut & Elevation) mit pvlib (NREL SPA).
    
    Args:
        lat: Breitengrad in Grad
        lon: Längengrad in Grad
        date: datetime object oder ISO string
        time_utc: Zeit-String "HH:MM" oder datetime
    
    Returns:
        (azimuth, elevation) in Grad
    """
    # Datum als datetime
    if isinstance(date, str):
        date = datetime.fromisoformat(date)
    
    # Zeit als datetime kombinieren
    if isinstance(time_utc, str):
        hour, minute = map(int, time_utc.split(':'))
        timestamp = date.replace(hour=hour, minute=minute)
    else:
        timestamp = time_utc
    
    # Als pandas Timestamp
    timestamp = pd.Timestamp(timestamp, tz='UTC')
    
    # Sonnenposition berechnen
    solar_position = pvlib.solarposition.get_solarposition(
        time=timestamp,
        latitude=lat,
        longitude=lon,
        method='nrel_numpy'  # NREL SPA Algorithm
    )
    
    azimuth = solar_position['azimuth'].iloc[0]
    elevation = solar_position['elevation'].iloc[0]
    
    return azimuth, elevation