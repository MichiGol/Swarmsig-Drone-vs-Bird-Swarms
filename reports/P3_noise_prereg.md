# P3-Nachtrag „Drohnenrauschen mit Wind“: Vorregistrierung

**Stand:** 2026-10-11, vor dem Lauf. Das Dokument wird vor dem Start committet.

## 1. Ziel

- **P1 ohne Wind:** Die Trennschärfe hängt stark am unabhängigen Prozessrauschen σ_d der Drohnen.
  Unabhängiger Seed 20261013, N = 50, D1 / D2 / D3:

  | σ_d | AUC′ ohne Wind |
  |---|---|
  | 0,1 | 0,85 / 0,89 / 0,99 |
  | 0,3 | 0,77 / 0,88 / 0,92 |
  | 1,0 | 0,54 / 0,78 / 0,72 |

- **P3 bei σ_d = 0,1:** Realer Herbstwind kostet nichts (0,88 / 0,88 / 0,99 mit Wind, 0,88 / 0,89 / 0,98 ohne).

**Frage:** Ändert der Wind die Empfindlichkeit gegenüber dem Drohnenrauschen? Bricht die Trennung
mit Wind bei hohem Rauschen stärker oder schwächer ein als ohne?

## 2. Design

- **Designpunkte:** dieselben 300 Designpunkte mit N = 50 wie im Hauptlauf `P3_wind_v1` (Seed 20261014,
  Indizes 600–899) mit denselben Zufallszahlen. Das Rauschen der Drohnen wird mit
  demselben Zufallsstrom erzeugt und nur skaliert.
- **Drohnen:** D1, D2, D3 mit σ_d = 0,3 (`configs/p3_noise_sd03.yaml`) und σ_d = 1,0
  (`configs/p3_noise_sd10.yaml`). σ_d = 0,1 liegt aus dem Hauptlauf vor.
- **Bedingungen:** Je σ_d mit Wind (K0) und windstill (K-Calm). Die Vögel stammen aus dem Hauptlauf (K0 bzw. K-Calm).
- **Übrige Einstellungen:** wie in `P3_wind_v1` (Wetter, Turbulenz, Windreaktion, Sensor, Statistik).

## 3. Auswertung (`scripts/analyze_p3_noise.py`)

- **Statistik:** AUC′ = P(T_nn,Drohne > T_nn,Vogel) mit Abzug A7, je σ_d, Regler sowie mit und ohne Wind.
  Designpunkt-Bootstrap mit 2000 Wiederholungen.
- **Gepaarte Differenz:** ΔAUC_Wind(σ_d) = AUC′(ohne Wind) − AUC′(mit Wind), mit gemeinsamen Bootstrap-Gewichten.
- **Lesart wie in P1, für σ_d = 1,0 mit Wind:**

  | Urteil | Bedingung |
  |---|---|
  | robust | alle drei Regler AUC′ ≥ 0,8 |
  | hängt wesentlich an der Rauschannahme | mindestens ein Regler < 0,7 |
  | teilweise robust | sonst |

- **Wechselwirkung Wind × Rauschen:** keine Richtungshypothese, die Auswertung ist beschreibend.
  Erwartung aus P3: Böen wirken überwiegend als gemeinsame Bewegung. Wind sollte die
  Rauschempfindlichkeit daher kaum ändern; erwartet ist |ΔAUC_Wind| ≤ 0,05 bei jedem σ_d.
- **Beschreibende Größen:** Mediane von T_nn und der Speed-Schwankung der Drohnen je σ_d, mit und ohne Wind.

## 4. Rechenzeit

| Umfang | Zeit |
|---|---|
| je σ_d (300 Designpunkte, 3 Drohnenregler mit und ohne Wind) | ≈ 7 min |
| zusammen | ≈ 14 min |

Das Zeitlimit liegt bei 12 min je Lauf. Die Projektleitung hat den Nachtrag am 2026-10-11 freigegeben
(Option 1 aus dem P3-Bericht, Abschnitt 9). Die gepaarte windstille Kontrolle kostet ≈ 4 min
zusätzlich.
