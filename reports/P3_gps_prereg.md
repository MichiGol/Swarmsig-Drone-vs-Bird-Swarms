# P3-Nachtrag „Positionierung: RTK gegen Standard-GPS“: Vorregistrierung

**Stand:** 2026-10-11, vor dem Lauf. Das Dokument wird vor dem Start committet.

## 1. Ausgangslage

Die Projektleitung hat eine Literaturübersicht zur Positionsgenauigkeit von Drohnen geliefert.
Die Kernzahlen sind geprüft: Ladino & Sama (2024, J. ASABE 67(6)) messen mit RTK 1–6 cm Fehler,
ohne RTK 50–315 cm.

Im bisherigen Modell (Architektur 4.7) wirkt GPS-Rauschen nur darauf, **wie eine Drohne ihre
Nachbarn sieht**. Den eigenen Zustand kennt sie exakt. Die vorhandenen P3-Daten zeigen, dass
dieses Nachbarrauschen der Unterscheidbarkeit nicht schadet: Eine Drohne sendet ihre fehlerhafte
Position an alle Nachbarn, die gemeinsam reagieren.

Zwei Dinge fehlen im Modell:

1. **Eigener Fehler:** Der eigene GPS-Fehler bewegt die Drohne selbst. Sie fliegt dorthin, wo sie zu sein glaubt.
2. **Gemeinsamer Fehler:** Die Fehler benachbarter Drohnen sind teilweise gleich, etwa durch Atmosphäre und Bahndaten.
   Unabhängig sind sie bei Empfängeruhr, Mehrwegeempfang und Rauschen. Die Literatur betont, dass sich bei getrennt
   berechneten Positionen viel, aber selten alles aufhebt.

## 2. Modellerweiterung (dokumentierte Abweichung von 4.7)

- **GPS-Fehler je Drohne:** ε_i = √(1−ρ)·ε_i^unabhängig + √ρ·ε^gemeinsam. Beide Teile sind OU-Prozesse mit
  gleichem σ_g und τ_g; für die Geschwindigkeit gilt dasselbe mit σ_gv. Bei ρ = 0 ergibt sich das Modell aus 4.7.
- **Eigener Zustand** (`own_state_noise`): Der Regler nutzt für sich selbst x_i + ε_i, u_i + η_i und
  im inneren Geschwindigkeitsregelkreis v_i + η_i.
- **Was wahr bleibt:** der Funkgraph (wer in Reichweite ist) und der für alle gleiche Navigationsterm.
- **Prüfung:** Numba gegen NumPy ist für alle Regler mit und ohne Wind getestet. Ohne Erweiterung rechnen die Kerne
  bit-genau wie zuvor.

## 3. Profile

| Parameter | RTK | Standard-GPS | Herkunft |
|---|---|---|---|
| σ_g (Position) | U[0,02; 0,06] m | U[1; 3] m | L: Ladino & Sama 2024; Herstellerangaben |
| σ_gv (Geschwindigkeit) | U[0,02; 0,06] m/s | U[0,05; 0,10] m/s | L/A: Doppler-Geschwindigkeit ≈ 0,05 m/s |
| τ_g (Zeitskala der Fehler) | U[1; 10] s | U[10; 60] s | A |
| ρ (gemeinsamer Anteil) | U[0; 0,5] | U[0,5; 0,9] | A |

Jedes Profil wird zweimal gerechnet, mit eigenem Fehler im Regler („mit“) und ohne („ohne“, Referenz).
Alle übrigen Parameter bleiben wie in `P3_wind_v1`, einschließlich σ_d = 0,1 und Wind.

## 4. Design

- **Designpunkte:** dieselben 300 Designpunkte mit N = 50 wie im Hauptlauf `P3_wind_v1` (Seed 20261014, Indizes 600–899)
  mit denselben Zufallszahlen.
- **Klassen:** Drohnen D1, D2, D3 mit Wind (K0). Die Vögel stammen aus dem Hauptlauf (K0).
- **Configs:** `configs/p3_pos_{rtk,gps}_{own,ref}.yaml`.

## 5. Auswertung (`scripts/analyze_p3_gps.py`)

- **Statistik:** AUC′ = P(T_nn,Drohne > T_nn,Vogel) mit Abzug A7, Designpunkt-Bootstrap mit 2000 Wiederholungen.
- **Primär:** Standard-GPS mit eigenem Fehler, also der praktisch wichtigste und kritischste Fall. Lesart je Profil
  wie in den früheren Sensitivitätsläufen:

  | Urteil | Bedingung |
  |---|---|
  | robust | alle drei Regler AUC′ ≥ 0,8 |
  | hängt wesentlich an der Positionierung | mindestens ein Regler < 0,7 |
  | teilweise robust | sonst |

- **Sekundär:**
  - RTK mit eigenem Fehler, gleiche Lesart.
  - Gepaarte Differenz Δ_eigen = AUC′(ohne) − AUC′(mit) je Profil und Regler, mit gemeinsamen Bootstrap-Gewichten.
  - Beschreibend: Mediane von T_nn und der Speed-Schwankung, Nächster-Nachbar-Abstand, Anteil zerfallener Schwärme.
- **Erwartung, ohne Richtungstest:**
  - RTK: Der eigene Fehler ist winzig, erwartet ist |Δ_eigen| ≤ 0,03.
  - Standard-GPS: Der unabhängige Fehleranteil von etwa 0,3–2 m wandert langsam (10–60 s). Das ergibt
    Geschwindigkeitsabweichungen von wenigen Zentimetern pro Sekunde und dazu den unabhängigen Teil des
    Geschwindigkeitsfehlers (≈ 0,02–0,07 m/s). Erwartet ist ein mäßiger Rückgang von AUC′, am ehesten bei D1 und D3.

## 6. Pilot (offengelegt)

- Umfang: je Profil mit eigenem Fehler die ersten 8 Designpunkte (Indizes 600–607), in einem Testordner gerechnet.
- Angesehen wurden nur Zusammenhalt, Abstände, Polarisation und Laufzeit; T_nn und AUC′ wurden nicht angesehen.
- Ergebnis: Die Schwärme halten zusammen, und Abstände wie Polarisation sind plausibel.

## 7. Rechenzeit

| Umfang | Zeit |
|---|---|
| je Config (300 Designpunkte, 3 Drohnenregler) | ≈ 6 min |
| zusammen (4 Configs) | ≈ 24 min |

Das Zeitlimit liegt bei 10 min je Config. Die Läufe starten mit `--keep-awake`. Die Projektleitung hat
Option A am 2026-10-11 freigegeben.
