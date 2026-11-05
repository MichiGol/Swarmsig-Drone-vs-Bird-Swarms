# O5 „Drohnen ohne Nachbarkopplung (Formation)“: Vorregistrierung

**Stand:** 2026-10-15. Das Dokument wird vor den Läufen committet.

## 1. Ziel

Bisher haben wir nur Drohnen mit Schwarmregler (D1–D3) betrachtet, die auf ihre Nachbarn reagieren. Viele echte
Drohnengruppen sind aber anders gesteuert:

- vorprogrammierte Formationen wie Lichtshows,
- Wegpunktflüge,
- einzeln gesteuerte Drohnen.

Jede Drohne folgt dabei ihrer eigenen Sollbahn, Nachbarn spielen keine Rolle.

**Frage:** Lassen sich solche Formationsdrohnen über die lokale Speed-Kopplung von Vögeln trennen?

## 2. Modell DF (`src/swarmsig/agents/drone.py`, Numba-Kern in `kernels.py`)

- **Regler:** v_cmd,i = u_r + k_form · (x_r + o_i − x̂_i), danach gesättigt.
  - x_r, u_r: gemeinsamer Referenzpunkt auf der Navigationsbahn (wie D2), Geschwindigkeit s0.
  - o_i: fester Versatz der Drohne zum Schwerpunkt der Startformation.
  - x̂_i: eigene Positionsschätzung.
  - Der innere Regelkreis ist wie bei D1/D3 (Zeitkonstante τ_v, Prozessrauschen σ_d, Wind, Realismus-Schicht).
  - **Kein Bezug zu Nachbarn, keine Kommunikation.**
- **Prior:** k_form ~ U[0,3; 1,5] 1/s (Annahme). Alle anderen Drohnenparameter wie bei D1–D3.
- **Tests:**
  - NumPy-Regler gleich Numba-Kern (Befehle und Bahnen, mit und ohne eigenen GPS-Fehler).
  - Ohne Rauschen hält DF die Startformation.
  - Ein verschobener Nachbar ändert keinen fremden Befehl.
- **Zufallsströme:** DF ist hinten an die Klassen-Registry angehängt, alle bisherigen Ströme bleiben unverändert.

## 3. Design

Dieselben 900 Designpunkte wie P3_wind_v1 und O3c_robust_v1: N = 20/30/50 × 300, Herbstwind, Kontrolle K0. Vögel und
D1–D3 stammen aus O3c_robust_v1.

| Variante | Konfiguration | Positionierung |
|---|---|---|
| base | `configs/o5_df_base.yaml` | wie P3-Hauptlauf: eigener Zustand exakt |
| rtk | `configs/o5_df_rtk.yaml` | eigener GPS-Fehler im Regler, RTK (2–6 cm), wie P3_pos_rtk_own |
| gps | `configs/o5_df_gps.yaml` | eigener GPS-Fehler im Regler, Standard-GPS (1–3 m), wie P3_pos_gps_own |

**Probelauf:** je 8 Designpunkte bei N = 20, nur Zeitmessung (≈ 0,9 s je Designpunkt). Angesehen habe ich nur
Abstand und Polarisation, keine Kopplungswerte.

**Rechenzeit:** etwa 4 min je Variante mit 4 Prozessen.

## 4. Auswertung (`scripts/analyze_o5.py`)

- **Kenngröße** (nach der Empfehlung aus O3c-1):
  - **primär Q bei N = 30 und 50, L bei N = 20**;
  - beide werden immer berichtet, dazu A7 und M als Referenz.
- **Hauptvergleich:** AUC′ = P(T_DF > T_Vogel) je Variante und N, Designpunkt-Bootstrap (2000).

  | Lesart | Bedingung |
  |---|---|
  | DF stärker gekoppelt (trennbar) | AUC′ ≥ 0,8 und Intervall über 0,5 |
  | schwache Tendenz zu DF | 0,5 < AUC′ < 0,8 und Intervall über 0,5 |
  | kein belastbarer Unterschied | Intervall enthält 0,5 |
  | Vögel stärker gekoppelt | Intervall unter 0,5 |

  Hauptaussage ist das Urteil bei N = 50 unter Q.
- **Beschreibend:**
  - AUC′ = P(T_D3 > T_DF), also Schwarmregler gegen Formation.
  - Mediane von T je Klasse.
  - Vergleich von DF mit den echten Tauben und Drohnen aus O3b.
- **Erwartung, ohne Richtungstest:**
  - Ohne Nachbarkopplung sollten die Abweichungen weitgehend unabhängig sein, also T nahe null und vogelähnlich.
  - Mit unabhängigen GPS-Fehlern noch mehr.
  - Gemeinsame Windböen werden durch die Abzüge entfernt, kleinräumige Turbulenz könnte etwas Kopplung erzeugen.
- **Bedeutung:** Fällt DF in den Bereich der Vögel, erkennt die Methode *Schwarmregler*, nicht Drohnen an sich. Das
  wäre eine wichtige Grenze für die Anwendung.

## 5. Ausgaben

- Ergebnisse: `reports/O5_results.md` und CSVs.
- Abbildung: `figures/fig20_formation.png`.
- Bericht: `reports/O5_Bericht.md`.

Die Projektleitung hat O5 am 2026-10-15 freigegeben.
