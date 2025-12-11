"""
Geolocation Solver - Stage 7

Berechnet mögliche Standorte basierend auf:
- Geschätzter Sonnenrichtung (Azimut, Elevation) aus Schatten-Analyse
- Datum und Uhrzeit
- Hemisphäre (Nord/Süd)

Verwendet NREL Solar Position Algorithm via pysolar.

MATHEMATIK:
- Sonnenazimut hängt hauptsächlich von der LOKALZEIT ab (→ Längengrad)
- Sonnenelevation hängt hauptsächlich vom BREITENGRAD ab
- Bei gegebenem UTC-Zeitpunkt und gemessenem Azimut/Elevation können wir
  den Standort eingrenzen
"""

import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

# pysolar für präzise Sonnenstandsberechnung
try:
    from pysolar.solar import get_altitude, get_azimuth
    PYSOLAR_AVAILABLE = True
except ImportError:
    PYSOLAR_AVAILABLE = False
    print("WARNING: pysolar not installed. Using simplified calculations.")


@dataclass
class GeoLocation:
    latitude: float   # -90 bis +90
    longitude: float  # -180 bis +180
    
    def __repr__(self):
        lat_dir = 'N' if self.latitude >= 0 else 'S'
        lon_dir = 'E' if self.longitude >= 0 else 'W'
        return f"{abs(self.latitude):.4f}°{lat_dir}, {abs(self.longitude):.4f}°{lon_dir}"


@dataclass 
class SunPosition:
    azimuth: float    # 0-360°, 0=Nord, 90=Ost, 180=Süd, 270=West
    elevation: float  # 0-90°, 0=Horizont, 90=Zenit


@dataclass
class GeolocationResult:
    success: bool
    locations: List[GeoLocation]  # Mögliche Standorte
    corridor: Dict  # Lat/Lon Bereiche
    confidence: float  # 0-100%
    sun_position_calculated: SunPosition
    sun_position_measured: SunPosition
    error_deg: float
    message: str
    details: Dict


# =============================================================================
# SONNENSTAND-BERECHNUNG
# =============================================================================

def get_sun_position(latitude: float, longitude: float, dt: datetime) -> SunPosition:
    """
    Berechne Sonnenstand für gegebene Koordinaten und Zeit.
    
    Args:
        latitude: Breitengrad (-90 bis +90)
        longitude: Längengrad (-180 bis +180)
        dt: datetime in UTC
        
    Returns:
        SunPosition mit Azimut (0-360°) und Elevation (0-90°)
    """
    if PYSOLAR_AVAILABLE:
        # pysolar verwendet: Azimut 0=Nord, positiv im Uhrzeigersinn
        # Elevation: positiv über Horizont
        
        # Stelle sicher, dass datetime UTC ist
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        elevation = get_altitude(latitude, longitude, dt)
        azimuth = get_azimuth(latitude, longitude, dt)
        
        # pysolar gibt Azimut manchmal negativ zurück, normalisieren auf 0-360
        azimuth = azimuth % 360
        
        return SunPosition(azimuth=azimuth, elevation=elevation)
    else:
        # Fallback: Vereinfachte Berechnung
        return _simplified_sun_position(latitude, longitude, dt)


def _simplified_sun_position(latitude: float, longitude: float, dt: datetime) -> SunPosition:
    """
    Verbesserte Sonnenstandsberechnung (ohne externe Libraries).
    Basiert auf NOAA Solar Calculator Algorithmen.
    Genauigkeit: ~0.5-1°
    """
    # Julian Day
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Konvertiere zu Julian Day Number
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    
    if month <= 2:
        year -= 1
        month += 12
    
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + hour/24 + B - 1524.5
    
    # Julian Century
    JC = (JD - 2451545) / 36525
    
    # Geometrische mittlere Länge der Sonne (Grad)
    L0 = (280.46646 + JC * (36000.76983 + 0.0003032 * JC)) % 360
    
    # Geometrische mittlere Anomalie der Sonne (Grad)
    M = (357.52911 + JC * (35999.05029 - 0.0001537 * JC)) % 360
    M_rad = math.radians(M)
    
    # Exzentrizität der Erdbahn
    e = 0.016708634 - JC * (0.000042037 + 0.0000001267 * JC)
    
    # Mittelpunktsgleichung der Sonne
    C = (math.sin(M_rad) * (1.914602 - JC * (0.004817 + 0.000014 * JC)) +
         math.sin(2 * M_rad) * (0.019993 - 0.000101 * JC) +
         math.sin(3 * M_rad) * 0.000289)
    
    # Wahre Länge der Sonne
    sun_true_long = L0 + C
    
    # Scheinbare Länge der Sonne
    omega = 125.04 - 1934.136 * JC
    sun_apparent_long = sun_true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    
    # Mittlere Schiefe der Ekliptik
    mean_obliq = 23 + (26 + (21.448 - JC * (46.8150 + JC * (0.00059 - JC * 0.001813))) / 60) / 60
    
    # Korrigierte Schiefe
    obliq_corr = mean_obliq + 0.00256 * math.cos(math.radians(omega))
    obliq_corr_rad = math.radians(obliq_corr)
    
    # Deklination der Sonne
    sun_decl = math.degrees(math.asin(math.sin(obliq_corr_rad) * 
                                       math.sin(math.radians(sun_apparent_long))))
    
    # Zeitgleichung (in Minuten)
    var_y = math.tan(obliq_corr_rad / 2) ** 2
    eq_of_time = 4 * math.degrees(
        var_y * math.sin(2 * math.radians(L0)) -
        2 * e * math.sin(M_rad) +
        4 * e * var_y * math.sin(M_rad) * math.cos(2 * math.radians(L0)) -
        0.5 * var_y * var_y * math.sin(4 * math.radians(L0)) -
        1.25 * e * e * math.sin(2 * M_rad)
    )
    
    # Wahre Sonnenzeit
    time_offset = eq_of_time + 4 * longitude  # in Minuten
    true_solar_time = (hour * 60 + dt.minute + dt.second/60 + time_offset) % 1440
    
    # Stundenwinkel
    if true_solar_time / 4 < 0:
        hour_angle = true_solar_time / 4 + 180
    else:
        hour_angle = true_solar_time / 4 - 180
    
    hour_angle_rad = math.radians(hour_angle)
    lat_rad = math.radians(latitude)
    decl_rad = math.radians(sun_decl)
    
    # Zenitwinkel
    cos_zenith = (math.sin(lat_rad) * math.sin(decl_rad) +
                  math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle_rad))
    cos_zenith = max(-1, min(1, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    
    # Elevation
    elevation = 90 - zenith
    
    # Azimut
    if cos_zenith > 0.99999:
        azimuth = 180  # Sonne im Zenit
    else:
        cos_azimuth = ((math.sin(lat_rad) * cos_zenith - math.sin(decl_rad)) /
                       (math.cos(lat_rad) * math.sin(math.radians(zenith))))
        cos_azimuth = max(-1, min(1, cos_azimuth))
        
        if hour_angle > 0:
            azimuth = (math.degrees(math.acos(cos_azimuth)) + 180) % 360
        else:
            azimuth = (540 - math.degrees(math.acos(cos_azimuth))) % 360
    
    return SunPosition(azimuth=azimuth, elevation=elevation)


# =============================================================================
# INVERSE BERECHNUNG: Sonnenstand → Koordinaten
# =============================================================================

def find_locations_for_sun_position(
    measured_azimuth: float,
    measured_elevation: float,
    dt: datetime,
    hemisphere: str = 'north',  # 'north' oder 'south'
    azimuth_tolerance: float = 5.0,
    elevation_tolerance: float = 5.0,
    grid_resolution: float = 0.5
) -> GeolocationResult:
    """
    Finde mögliche Standorte für gemessenen Sonnenstand.
    
    Args:
        measured_azimuth: Gemessener Sonnen-Azimut (0-360°)
        measured_elevation: Gemessene Sonnen-Elevation (0-90°)
        dt: Datum/Uhrzeit in UTC
        hemisphere: 'north' oder 'south'
        azimuth_tolerance: Toleranz für Azimut in Grad
        elevation_tolerance: Toleranz für Elevation in Grad
        grid_resolution: Auflösung des Suchgitters in Grad
        
    Returns:
        GeolocationResult mit möglichen Standorten
    """
    if measured_elevation <= 0:
        return GeolocationResult(
            success=False,
            locations=[],
            corridor={},
            confidence=0,
            sun_position_calculated=SunPosition(0, 0),
            sun_position_measured=SunPosition(measured_azimuth, measured_elevation),
            error_deg=0,
            message="Sonnenelevation muss positiv sein (Sonne über Horizont)",
            details={}
        )
    
    # Breitengrad-Bereich basierend auf Hemisphäre
    if hemisphere == 'north':
        lat_range = (0, 72)  # Nicht bis 90° (Polarnacht/Mitternachtssonne)
    else:
        lat_range = (-72, 0)
    
    # Längengrad: voller Bereich
    lon_range = (-180, 180)
    
    # Grobsuche: Finde Regionen mit passendem Sonnenstand
    matching_points = []
    
    # Erste Phase: Grobes Gitter (2°)
    coarse_resolution = 2.0
    for lat in np.arange(lat_range[0], lat_range[1], coarse_resolution):
        for lon in np.arange(lon_range[0], lon_range[1], coarse_resolution):
            sun = get_sun_position(lat, lon, dt)
            
            # Prüfe ob Sonnenstand passt (mit größerer Toleranz für Grobsuche)
            az_error = min(abs(sun.azimuth - measured_azimuth),
                          360 - abs(sun.azimuth - measured_azimuth))
            el_error = abs(sun.elevation - measured_elevation)
            
            if az_error < azimuth_tolerance * 2 and el_error < elevation_tolerance * 2:
                matching_points.append((lat, lon, az_error, el_error))
    
    if not matching_points:
        return GeolocationResult(
            success=False,
            locations=[],
            corridor={},
            confidence=0,
            sun_position_calculated=SunPosition(0, 0),
            sun_position_measured=SunPosition(measured_azimuth, measured_elevation),
            error_deg=0,
            message=f"Kein Standort gefunden für Azimut {measured_azimuth:.1f}°, Elevation {measured_elevation:.1f}° zur angegebenen Zeit",
            details={'searched_hemisphere': hemisphere}
        )
    
    # Zweite Phase: Feinsuche um die besten Grobpunkte
    refined_points = []
    
    # Sortiere nach Gesamtfehler und nimm die besten Regionen
    matching_points.sort(key=lambda x: x[2] + x[3])
    best_regions = matching_points[:20]  # Top 20 Regionen
    
    for lat_center, lon_center, _, _ in best_regions:
        # Feinsuche in 0.5° Auflösung um diesen Punkt
        for lat in np.arange(lat_center - 2, lat_center + 2, grid_resolution):
            if lat < lat_range[0] or lat > lat_range[1]:
                continue
            for lon in np.arange(lon_center - 2, lon_center + 2, grid_resolution):
                # Längengrad wrappen
                lon_wrapped = ((lon + 180) % 360) - 180
                
                sun = get_sun_position(lat, lon_wrapped, dt)
                
                az_error = min(abs(sun.azimuth - measured_azimuth),
                              360 - abs(sun.azimuth - measured_azimuth))
                el_error = abs(sun.elevation - measured_elevation)
                
                if az_error < azimuth_tolerance and el_error < elevation_tolerance:
                    total_error = math.sqrt(az_error**2 + el_error**2)
                    refined_points.append({
                        'lat': lat,
                        'lon': lon_wrapped,
                        'az_error': az_error,
                        'el_error': el_error,
                        'total_error': total_error,
                        'sun': sun
                    })
    
    if not refined_points:
        return GeolocationResult(
            success=False,
            locations=[],
            corridor={},
            confidence=0,
            sun_position_calculated=SunPosition(0, 0),
            sun_position_measured=SunPosition(measured_azimuth, measured_elevation),
            error_deg=0,
            message="Kein exakter Standort gefunden (Toleranz zu eng?)",
            details={'coarse_matches': len(matching_points)}
        )
    
    # Sortiere nach Fehler
    refined_points.sort(key=lambda x: x['total_error'])
    
    # Beste Punkte als Locations
    best_point = refined_points[0]
    locations = [
        GeoLocation(p['lat'], p['lon']) 
        for p in refined_points[:10]  # Top 10
    ]
    
    # Berechne Korridor (Bounding Box der guten Punkte)
    good_points = [p for p in refined_points if p['total_error'] < azimuth_tolerance]
    if good_points:
        lats = [p['lat'] for p in good_points]
        lons = [p['lon'] for p in good_points]
        corridor = {
            'lat_min': min(lats),
            'lat_max': max(lats),
            'lon_min': min(lons),
            'lon_max': max(lons),
            'lat_center': np.mean(lats),
            'lon_center': np.mean(lons)
        }
    else:
        corridor = {
            'lat_min': best_point['lat'] - 1,
            'lat_max': best_point['lat'] + 1,
            'lon_min': best_point['lon'] - 1,
            'lon_max': best_point['lon'] + 1,
            'lat_center': best_point['lat'],
            'lon_center': best_point['lon']
        }
    
    # Confidence basierend auf Fehler
    confidence = max(0, 100 - best_point['total_error'] * 10)
    
    return GeolocationResult(
        success=True,
        locations=locations,
        corridor=corridor,
        confidence=confidence,
        sun_position_calculated=best_point['sun'],
        sun_position_measured=SunPosition(measured_azimuth, measured_elevation),
        error_deg=best_point['total_error'],
        message=f"Standort gefunden: {locations[0]} (±{best_point['total_error']:.2f}°)",
        details={
            'total_matches': len(refined_points),
            'best_azimuth_error': best_point['az_error'],
            'best_elevation_error': best_point['el_error'],
            'search_grid_resolution': grid_resolution,
            'hemisphere': hemisphere
        }
    )


def calculate_geolocation(
    session_data: Dict,
    screenshot_id: str,
    date_str: str,  # "YYYY-MM-DD"
    time_str: str,  # "HH:MM" in UTC
    hemisphere: str = 'north',
    room_orientation: float = 0.0  # In welche Himmelsrichtung zeigt die Front-Wand (z=0)?
) -> Dict:
    """
    Hauptfunktion: Berechne Geolocation für einen Screenshot.
    
    Verwendet die Lichtrichtung aus der Validierung und berechnet
    mögliche Standorte.
    
    Args:
        session_data: Session-Daten mit Kalibrierung und Schatten
        screenshot_id: ID des Screenshots
        date_str: Datum "YYYY-MM-DD"
        time_str: Uhrzeit "HH:MM" in UTC
        hemisphere: 'north' oder 'south'
        room_orientation: In welche Himmelsrichtung zeigt die Front-Wand (z=0)?
                         0=Nord, 90=Ost, 180=Süd, 270=West
                         WICHTIG: Die Raumausrichtung im 3D-Kalibrierer ist willkürlich!
                         Dieser Wert muss die ECHTE Ausrichtung zur Welt angeben.
    """
    # 1. Finde Schatten-Daten für diesen Screenshot
    shadows = session_data.get('shadows', [])
    screenshot_shadows = None
    for s in shadows:
        if s.get('screenshotId') == screenshot_id:
            screenshot_shadows = s
            break
    
    if not screenshot_shadows:
        return {
            'success': False,
            'message': f'Keine Schatten-Daten für Screenshot {screenshot_id}'
        }
    
    # 2. Validiere die Objekte um Lichtrichtung zu bekommen
    # Import hier um zirkuläre Imports zu vermeiden und flexibel zu sein
    try:
        from solver.validation import validate_inter_object
    except ImportError:
        try:
            from .validation import validate_inter_object
        except ImportError:
            # Fallback: versuche direkt aus demselben Verzeichnis
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from validation import validate_inter_object
    
    calibration = session_data.get('calibration', {})
    
    # Finde Screenshot-Kalibrierung
    screenshot_calib = None
    for sc in calibration.get('screenshots', []):
        if sc.get('screenshotId') == screenshot_id:
            screenshot_calib = sc
            break
    
    if not screenshot_calib:
        return {
            'success': False,
            'message': f'Keine Kalibrierung für Screenshot {screenshot_id}'
        }
    
    # Validiere um Lichtrichtung zu bekommen
    validation_result = validate_inter_object(
        screenshot_shadows.get('objects', []),
        calibration.get('camera', {}).get('position', {}),
        screenshot_calib.get('cameraRotation', {}),
        calibration.get('camera', {}).get('fovY', 60),
        calibration.get('room', {})
    )
    
    if not validation_result.get('success'):
        return {
            'success': False,
            'message': 'Validierung fehlgeschlagen: ' + validation_result.get('message', '')
        }
    
    # 3. Extrahiere Sonnenrichtung
    # WICHTIG: Die Lichtrichtung zeigt VON der Sonne ZUM Objekt
    # Wir brauchen die Richtung ZUR Sonne (umgekehrt)
    mean_azimuth = validation_result.get('mean_light_azimuth_deg', 0)
    mean_elevation = validation_result.get('mean_light_elevation_deg', 0)
    
    # Azimut umkehren (180° drehen) da wir Richtung ZUR Sonne brauchen
    sun_azimuth_model = (mean_azimuth + 180) % 360
    
    # RAUMAUSRICHTUNG ANWENDEN: Modell-Koordinaten → Welt-Koordinaten
    # room_orientation gibt an, wohin +Z zeigt (0=Nord, 90=Ost, etc.)
    sun_azimuth = (sun_azimuth_model + room_orientation) % 360
    
    sun_elevation = mean_elevation  # Elevation bleibt gleich
    
    # 4. Parse Datum/Uhrzeit
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        dt = dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        return {
            'success': False,
            'message': f'Ungültiges Datum/Uhrzeit Format: {e}'
        }
    
    # 5. Finde mögliche Standorte
    result = find_locations_for_sun_position(
        measured_azimuth=sun_azimuth,
        measured_elevation=sun_elevation,
        dt=dt,
        hemisphere=hemisphere,
        azimuth_tolerance=5.0,
        elevation_tolerance=3.0,
        grid_resolution=0.25
    )
    
    # 6. Formatiere Ergebnis
    return {
        'success': result.success,
        'message': result.message,
        'data': {
            'locations': [
                {'latitude': loc.latitude, 'longitude': loc.longitude}
                for loc in result.locations
            ],
            'corridor': result.corridor,
            'confidence': result.confidence,
            'sun_position': {
                'measured_azimuth': sun_azimuth,
                'measured_elevation': sun_elevation,
                'calculated_azimuth': result.sun_position_calculated.azimuth,
                'calculated_elevation': result.sun_position_calculated.elevation
            },
            'shadow_analysis': {
                'light_azimuth': mean_azimuth,
                'light_elevation': mean_elevation,
                'sun_azimuth_model': sun_azimuth_model,
                'room_orientation': room_orientation,
                'inter_object_score': validation_result.get('inter_object_score', 0)
            },
            'input': {
                'date': date_str,
                'time_utc': time_str,
                'hemisphere': hemisphere,
                'room_orientation': room_orientation
            },
            'error_deg': result.error_deg
        } if result.success else None
    }


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def format_coordinates_dms(lat: float, lon: float) -> str:
    """Formatiere Koordinaten in Grad-Minuten-Sekunden."""
    def to_dms(deg, is_lat):
        direction = ('N' if deg >= 0 else 'S') if is_lat else ('E' if deg >= 0 else 'W')
        deg = abs(deg)
        d = int(deg)
        m = int((deg - d) * 60)
        s = ((deg - d) * 60 - m) * 60
        return f"{d}°{m}'{s:.1f}\"{direction}"
    
    return f"{to_dms(lat, True)}, {to_dms(lon, False)}"


def get_google_maps_url(lat: float, lon: float, zoom: int = 10) -> str:
    """Generiere Google Maps URL für Koordinaten."""
    return f"https://www.google.com/maps?q={lat},{lon}&z={zoom}"


def get_openstreetmap_url(lat: float, lon: float, zoom: int = 10) -> str:
    """Generiere OpenStreetMap URL für Koordinaten."""
    return f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map={zoom}/{lat}/{lon}"


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GEOLOCATION SOLVER TEST")
    print("=" * 60)
    
    if not PYSOLAR_AVAILABLE:
        print("⚠ pysolar nicht installiert - verwende vereinfachte Berechnung")
    else:
        print("✓ pysolar verfügbar")
    
    # Test: Bekannter Standort
    # Dresden: 51.05°N, 13.74°E
    test_lat = 51.05
    test_lon = 13.74
    test_time = datetime(2025, 6, 21, 12, 0, 0, tzinfo=timezone.utc)  # Sommersonnenwende, Mittag UTC
    
    print(f"\nTest-Standort: {test_lat}°N, {test_lon}°E")
    print(f"Test-Zeit: {test_time}")
    
    sun = get_sun_position(test_lat, test_lon, test_time)
    print(f"Berechneter Sonnenstand: Azimut={sun.azimuth:.1f}°, Elevation={sun.elevation:.1f}°")
    
    # Inverse Suche
    print("\n--- Inverse Suche ---")
    result = find_locations_for_sun_position(
        measured_azimuth=sun.azimuth,
        measured_elevation=sun.elevation,
        dt=test_time,
        hemisphere='north',
        azimuth_tolerance=2.0,
        elevation_tolerance=2.0,
        grid_resolution=0.25
    )
    
    print(f"Erfolg: {result.success}")
    print(f"Nachricht: {result.message}")
    print(f"Confidence: {result.confidence:.1f}%")
    print(f"Fehler: {result.error_deg:.2f}°")
    
    if result.locations:
        best = result.locations[0]
        print(f"\nBester Treffer: {best}")
        print(f"Abweichung vom Original: {abs(best.latitude - test_lat):.2f}° Lat, {abs(best.longitude - test_lon):.2f}° Lon")
        print(f"\nGoogle Maps: {get_google_maps_url(best.latitude, best.longitude)}")
