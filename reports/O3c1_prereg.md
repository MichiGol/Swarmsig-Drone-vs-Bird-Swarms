# O3c-1 „Simulation unter robusten Kenngrößen“: Vorregistrierung

**Stand:** 2026-10-15. Das Dokument wird vor dem Lauf committet.

## 1. Ziel

O3b hat gezeigt:

- A7 und M verfälschen die Nachbarkopplung schon bei leichten Kurven bzw. Drehungen.
- Unter den robusten Kenngrößen Q (primär), L und RL sind echte Tauben ≈ 0, echte Drohnen ≈ 0,1 und echte Dohlen ≈ 0,3 gekoppelt.

Die Simulation kennt diese Kenngrößen noch nicht. Daraus folgen zwei Fragen:

1. **Hält der simulierte Befund „Drohnen stärker gekoppelt als Vögel“ (P1/P3) unter Q, L und RL?**
2. **Treffen unsere Modelle die echten Daten unter Q, L und RL?** Das betrifft Vögel gegen Tauben bzw. Dohlen und
   D3 gegen die echten Vásárhelyi-Drohnen (gleicher Regler).

## 2. Design

- **Konfiguration:** `configs/o3c_robust.yaml`, dieselben 900 Designpunkte wie `P3_wind_v1` (N = 20/30/50 × 300,
  mit realem Herbstwind, Kontrolle K0) und dieselben Zufallszahlen.
- **Kenngrößen:** A7, M, A12 (unverändert) sowie L, Q, RL.
- **Bias-Tabelle:** `data/processed/nn_bias_table_robust.parquet`.
  - A7/M/A12 stammen aus der bisherigen Tabelle, L/Q/RL sind neu (2000 Wiederholungen je N).
  - Q ist bei N = 10 in 3D entartet (10 Parameter). Die Simulation nutzt nur N ≥ 20.
- **Probelauf (vor diesem Dokument angesehen):**
  - 8 Designpunkte bei N = 20, nur als Zeit- und Reproduktionsprüfung.
  - Ergebnis: A7, M und A12 stimmen exakt mit P3 überein, etwa 3,6 s je Designpunkt.
  - Gesehen habe ich auch die Einzelwerte. Unter Q lag D3 dort meist nahe 0. Das ist zu wenig für eine Aussage, wird aber
    der Ehrlichkeit halber genannt.
- **Rechenzeit:** etwa 22 min mit 4 Prozessen, `--keep-awake`.

## 3. Auswertung (`scripts/analyze_o3c.py`)

**(0) Reproduktion:** T_nn unter A7, M und A12 muss für alle Zeilen exakt P3_wind_v1 (K0) gleichen.

**(1) Simulierter Befund unter robusten Kenngrößen**

- AUC′ = P(T_Drohne > T_Vogel) je Drohnenklasse D1/D2/D3 und je N.
- Primär unter Q, ergänzend unter L und RL, zum Vergleich A7.
- Designpunkt-Bootstrap mit 2000 Wiederholungen (wie P3, ganze Designpunkte).

| Lesart | Bedingung |
|---|---|
| Drohnen stärker gekoppelt | AUC′ ≥ 0,8 und Intervall über 0,5 |
| schwache Tendenz zu Drohnen | 0,5 < AUC′ < 0,8 und Intervall über 0,5 |
| kein belastbarer Unterschied | Intervall enthält 0,5 |
| Vögel stärker gekoppelt | Intervall unter 0,5 |

Hauptaussage ist das Urteil bei N = 50 (wie P3). N = 20 und 30 werden mitberichtet. Bei N = 20 behält Q laut
Test-Suite wenig echte Kopplung, dort gelten L/RL als gleichwertig.

**(2) Realismus gegen echte Daten** (unter Q, L, RL; echte Werte aus `reports/O3b_echte_Voegel_units.csv`)

| echte Daten | Simulation |
|---|---|
| Tauben, kleine Schwärme (Kontrolle, 2D, q = 5) | Vögel B1/B3, N = 20 |
| Tauben, große Schwärme | Vögel, N = 30 |
| Dohlen (ganze Schwärme) | Vögel, N = 50 |
| echte Drohnen (3D, q = 5; ≈ 29 Drohnen) | D3, N = 30 |

| Lesart | Bedingung |
|---|---|
| realistisch | echter Median im 10–90-%-Bereich der Simulation |
| Modell zu schwach gekoppelt | echter Median über dem 90-%-Punkt |
| Modell zu stark gekoppelt | echter Median unter dem 10-%-Punkt |

Simulierte Werte stammen von einem idealen Sensor, echte enthalten Messrauschen. Das senkt die echte Kopplung; die
Lesart „Modell zu stark gekoppelt“ ist deshalb nur belastbar, wenn sie auch unter L und RL gilt.

**Erwartung, ohne Richtungstest:** Nach dem Probelauf könnte D3 unter Q deutlich schwächer gekoppelt sein als unter A7.

## 4. Ausgaben

- Ergebnisse: `reports/O3c1_results.md` und CSVs.
- Abbildung: `figures/fig19_simulation_robust.png`.
- Bericht: `reports/O3c1_Bericht.md`.
- Abweichungen werden im Bericht einzeln genannt.

Die Projektleitung hat den Lauf am 2026-10-15 freigegeben.
