import numpy as np
import asyncio
from typing import AsyncGenerator
from datetime import datetime, timedelta
from .sun_position import calculate_sun_position

async def solve_location_async(
    shadow_data: list,
    constraints: dict,
    settings: dict
) -> AsyncGenerator[dict, None]:
    """
    Asynchrone Standort-Berechnung mit Progress-Updates.
    Yields Progress-Dicts für WebSocket.
    """
    
    # Phase 1: Grobes Grid
    yield {
        "type": "progress",
        "phase": "Grobes Grid durchsuchen",
        "percent": 0,
        "solutionsFound": 0
    }
    
    lat_range = constraints['lat_range']
    lon_range = constraints['lon_range']
    
    # Grobes Raster: 2° Schritte
    lat_grid = np.arange(lat_range[0], lat_range[1], 2.0)
    lon_grid = np.arange(lon_range[0], lon_range[1], 2.0)
    
    candidates = []
    total_points = len(lat_grid) * len(lon_grid) * 36 * 14
    evaluated = 0
    
    # Datum-Range (vereinfacht - später aus constraints)
    date_start = datetime(2024, 3, 1)
    dates = [date_start + timedelta(days=i*10) for i in range(36)]
    
    # Zeit-Range
    times = [f"{h:02d}:00" for h in range(6, 20)]
    
    for lat in lat_grid:
        for lon in lon_grid:
            for date in dates:
                for time_str in times:
                    # Evaluiere diese Kombination
                    error = evaluate_location(
                        lat, lon, date, time_str, shadow_data
                    )
                    
                    if error < 10.0:  # Threshold
                        candidates.append({
                            'lat': float(lat),
                            'lon': float(lon),
                            'date': date.isoformat(),
                            'time': time_str,
                            'error': float(error)
                        })
                    
                    evaluated += 1
                    
                    # Progress-Update alle 1000 Evaluationen
                    if evaluated % 1000 == 0:
                        yield {
                            "type": "progress",
                            "phase": "Grobes Grid durchsuchen",
                            "percent": int(evaluated / total_points * 50),
                            "solutionsFound": len(candidates)
                        }
                        await asyncio.sleep(0)  # Yield control
    
    # Phase 2: Verfeinerung
    yield {
        "type": "progress",
        "phase": "Kandidaten verfeinern",
        "percent": 50,
        "solutionsFound": len(candidates)
    }
    
    refined_solutions = []
    for i, candidate in enumerate(candidates):
        # Lokale Optimierung (vereinfacht)
        refined = refine_candidate(candidate, shadow_data)
        refined_solutions.append(refined)
        
        if i % 5 == 0:
            yield {
                "type": "progress",
                "phase": "Kandidaten verfeinern",
                "percent": 50 + int(i / len(candidates) * 40),
                "solutionsFound": len(refined_solutions)
            }
            await asyncio.sleep(0)
    
    # Phase 3: Fertig
    yield {
        "type": "progress",
        "phase": "Abschließen",
        "percent": 90,
        "solutionsFound": len(refined_solutions)
    }
    
    # Sortiere nach Error
    refined_solutions.sort(key=lambda s: s['error'])
    
    # Nur Top-N behalten
    max_solutions = settings.get('maxSolutions', 200)
    final_solutions = refined_solutions[:max_solutions]
    
    # Finale Ergebnisse
    yield {
        "type": "complete",
        "percent": 100,
        "solutions": final_solutions,
        "statistics": {
            "totalEvaluations": evaluated,
            "candidatesFound": len(candidates),
            "uniqueSolutions": len(final_solutions)
        }
    }

def evaluate_location(lat, lon, date, time_str, shadow_data):
    """
    Berechnet Error für eine Standort-Hypothese.
    
    TODO: Hier kommt die echte Berechnung mit SPA rein.
    Aktuell: Dummy für Testing.
    """
    total_error = 0.0
    
    for measurement in shadow_data:
        # Zeit für diese Messung
        time_offset = measurement['time_offset_seconds']
        
        # Hier würde die echte Sonnenpositions-Berechnung stehen:
        # sun_az, sun_el = calculate_sun_position(lat, lon, date, time + offset)
        # Dann Vergleich mit gemessenen Schatten-Vektoren
        
        # Dummy für jetzt
        total_error += np.random.uniform(0, 5)
    
    return total_error

def refine_candidate(candidate, shadow_data):
    """
    Lokale Optimierung um Kandidaten.
    
    TODO: Hier würde Gradient Descent o.ä. reinkommen.
    Aktuell: Dummy.
    """
    # Füge Unsicherheit hinzu
    candidate['uncertainty'] = 0.05  # ±50m in Grad
    candidate['location_name'] = f"Location {candidate['lat']:.2f}°N, {candidate['lon']:.2f}°E"
    
    return candidate