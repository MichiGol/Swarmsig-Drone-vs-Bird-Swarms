# P3-Nachtrag „Messrauschen des Sensors“ (O2): Vorregistrierung

**Stand:** 2026-10-12, vor dem Lauf. Das Dokument wird vor dem Start committet.

## 1. Ziel

Bisher beobachtet ein *idealer* Sensor die Schwärme: Positionen ohne Fehler, 10 Bilder pro Sekunde,
Geschwindigkeit aus zentralen Differenzen über ±q Bilder. Echte Tracker haben Positionsfehler. Die
Geschwindigkeitsschätzung aus Positionsdifferenzen verstärkt diese Fehler:

    σ_v = √2 · σ_pos / (2 q δ),   δ = 0,1 s

| σ_pos | σ_v bei q = 1 (±0,1 s) | σ_v bei q = 5 (±0,5 s) |
|---|---|---|
| 1 cm | 0,07 m/s | 0,014 m/s |
| 5 cm | 0,35 m/s | 0,07 m/s |
| 20 cm | 1,41 m/s | 0,28 m/s |

Dieses Messrauschen ist für jedes Individuum unabhängig. Nach dem bisher verstandenen Mechanismus
(unabhängige Schwankungen senken die Nachbarkorrelation) bedroht es vor allem die ruhigen Drohnen
D1 und D3, deren eigene Speed-Schwankungen nur 0,1–0,2 m/s betragen.

**Frage:** Bleibt die Unterscheidbarkeit bei realistischem Messrauschen erhalten, und hilft ein
längeres Zeitfenster der Geschwindigkeitsschätzung?

## 2. Design

- **Designpunkte:** dieselben 300 Designpunkte mit N = 50 wie im Hauptlauf `P3_wind_v1` (Seed 20261014,
  Indizes 600–899, mit realem Herbstwind), mit denselben Zufallszahlen.
- **Sensorvarianten:** Jede simulierte Flugbahn wird mit 8 Sensoreinstellungen ausgewertet,
  σ_pos ∈ {0; 0,01; 0,05; 0,2} m × q ∈ {1; 5}. Vögel und Drohnen sehen jeweils denselben Sensor.
  Die Variante σ_pos = 0, q = 1 ist die Grundauswertung und reproduziert `P3_wind_v1` exakt
  (geprüft an 8 Designpunkten).
- **Rauschen:** Das Positionsrauschen kommt je Variante aus einem eigenen Zufallsstrom.
- **Umsetzung:** `configs/p3_sensor.yaml`, Sensorvarianten in `experiment/run.py`
  (`sensor.variants`). Getestet: Die Grundzeile bleibt bit-gleich, und die rauschfreie Variante gleicht ihr.

## 3. Auswertung (`scripts/analyze_p3_sensor.py`)

- **Statistik:** AUC′ = P(T_nn,Drohne > T_nn,Vogel) mit Abzug A7 je Variante und Regler, Designpunkt-Bootstrap mit
  2000 Wiederholungen.
- **Gepaarte Differenz:** Δ = AUC′(Grundauswertung) − AUC′(Variante), mit gemeinsamen Bootstrap-Gewichten.
- **Lesart je Variante:**

  | Urteil | Bedingung |
  |---|---|
  | robust | alle drei Regler AUC′ ≥ 0,8 |
  | hängt wesentlich am Messrauschen | mindestens ein Regler < 0,7 |
  | teilweise robust | sonst |

- **Referenzniveau:** σ_pos = 0,05 m, eine Variante aus Architektur 9.6. Berichtet wird es für beide q, keines
  wird vorab bevorzugt.
- **Erwartung, ohne Richtungstest:**
  - T_nn sinkt etwa um den Faktor Var_echt/(Var_echt + σ_v²).
  - Bei q = 1 und σ_pos ≥ 0,05 m erwarten wir einen starken Rückgang für D1/D3 bis hin zur Umkehr, für Vögel einen milden.
  - Bei q = 5 erwarten wir einen deutlich kleineren Rückgang.
  - Die Variante σ_pos = 0, q = 5 trennt die Glättung der echten Schwankungen vom Rauscheffekt.
- **Beschreibend:** Mediane von T_nn und der Speed-Schwankung je Klasse und Variante.

## 4. Rechenzeit und Pilot

- **Rechenzeit:** etwa 15 min mit 4 Prozessen. Das Zeitlimit liegt bei 25 min, gestartet wird mit `--keep-awake`
  im Wachbetrieb.
- **Pilot:** 8 Designpunkte (Indizes 600–607) für Zeitmessung und Reproduktionsprüfung. Angesehen wurde nur der
  Abgleich der Grundzeilen mit `P3_wind_v1` (identisch), nicht die Varianten.

Die Projektleitung hat O2 am 2026-10-12 freigegeben.
