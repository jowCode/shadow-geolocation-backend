"""
Geolocation Solver Package

Enthält:
- geolocation.py: Hauptlogik für Standortberechnung aus Sonnenstand
- geolocation_api.py: FastAPI Endpoint-Code zum Kopieren
"""

from .geolocation import (
    calculate_geolocation,
    find_locations_for_sun_position,
    get_sun_position,
    GeoLocation,
    SunPosition,
    GeolocationResult
)

__all__ = [
    'calculate_geolocation',
    'find_locations_for_sun_position', 
    'get_sun_position',
    'GeoLocation',
    'SunPosition',
    'GeolocationResult'
]
