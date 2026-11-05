# P1-Vorregistrierung: kleine Schwärme (P1_small_v1)

Festgeschrieben am 2026-10-11 vor dem Start der Hauptrechnung. Die Entscheidung wurde von der
Projektleitung an die Umsetzung delegiert („ich vertraue dir“). Änderungen nach dem Start
werden als Abweichung berichtet.

| Punkt | Festlegung |
|---|---|
| Config | `configs/p1_small.yaml` (Basis `base.yaml`), Seed 20261012 |
| Gruppengrößen, Designpunkte | N = 10, 20, 30, 50 je 300; N = 100: 150 (Grenzfall, siehe unten). Reihenfolge aufsteigend nach N, Zeitlimit 20 min; nicht gestartete Punkte werden als fehlend berichtet. |
| Klassen | Vögel: B1 oder B3 je Designpunkt (50/50); B3 mit Speed-Regelung steif/marginal und Nachbarschaft topologisch/metrisch (je 50/50). Drohnen: D1, D2 (Reichweite lokal, r = 1,2·d), D3, je Designpunkt alle drei. |
| Sensor | ideal, σ_pos = 0, 10 Hz, T_obs = 10 s nach 30 s Einschwingen; kein Wind |
| Primärstatistik | **T_nn mit Pflichtabzug A7** (7.4/7.5; Biastabelle `nn_bias_table_small.parquet`) |
| Sekundärstatistiken | T_s, T_⊥, T_χ (A7); alle Statistiken zusätzlich mit Abzug M und A12 |
| Entscheidungsrichtung | „Vögel“, falls T > τ (Erwartung: marginale Speed-Regelung → größere Nachbarkorrelation) |
| Schätzgröße | AUC je (N, Drohnenregler) als U-Statistik ohne Diagonale; 95-%-Intervall per Designpunkt-Bootstrap (2000 Resamples, Perzentil) |
| Go | In mindestens einem N: AUC ≥ 0,8 **für alle drei Drohnenregler**, jeweils mit unterer Intervallgrenze > 0,7 |
| No-go | AUC < 0,7 für alle N und alle Regler |
| Berichtsschnitte | je N und Regler; getrennt nach Vogelmodell (B1, B3-steif, B3-marginal); D2/D1/D3 einzeln |
| Ausschlüsse | keine. Läufe mit Zerfall (`frag_clusters > 1`) oder Instationarität bleiben drin; zusätzlich wird eine Auswertung nur mit zusammenhängenden Schwärmen berichtet. |
| Bekannte Grenzen | Vogelmodell bei N = 50 kalibriert (Light-ABC); bei N ≥ 100 Zerfall möglich (Pilot: N = 300 zerfällt). Zeitschritt je Vogel-Parametersatz aus der Kalibrierung (Regel 4.9); Drohnen 10 ms. |

Abweichungen vom Architekturdokument (Begründung im Pilotbericht `reports/P1_pilot.md`):
D1-Vorfaktor τ_v statt Δt_c; D2 nur lokale Reichweite; D3 mit SwarmLab-Anziehung; Drohnen
höchstens 10 Nachbarn; Vögel: Zusammenhalt über n_coh = 10–30 nächste Nachbarn (kalibriert);
Einschwingzeit 30 s statt 60 s; N = 200/300 sowie D2-c, Radar und Wind entfallen.
