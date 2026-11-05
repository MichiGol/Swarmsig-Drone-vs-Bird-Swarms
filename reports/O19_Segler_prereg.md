# O19 „Schornsteinsegler als dritte echte Vogelart“: Vorregistrierung

**Stand:** 2026-10-16. Das Dokument wird vor der Auswertung committet.

## Frage

Unter den robusten Kenngrößen sind echte Tauben praktisch ungekoppelt (Q ≈ 0), echte Dohlen deutlich gekoppelt
(Q ≈ 0,29, L ≈ 0,48), echte Drohnen liegen dazwischen (Q ≈ 0,10, L ≈ 0,19). **Wohin fallen Schornsteinsegler**, eine
dritte Art in großen Schwärmen?

## Daten

- **Quelle:** Evangelista, Ray, Raja & Hedrick (2017), Proc. R. Soc. B, doi:10.1098/rspb.2016.2602; Zenodo 5082229, CC0.
  Ausgewertet wird die Datei `A_nodes.hdf5` (md5 geprüft).
- **Inhalt:** Stereokameras mit 29,97 Bildern/s, Positionen und von den Autoren geglättete Geschwindigkeiten in 3D;
  nur Bahnen ab 45 Bildern.
- **Situation:** Schwarm kreist vor dem Schlafplatz (Kamin) in Raleigh, NC.
- **Vorab angesehen** (nur Aufbau, keine Kopplungswerte): 15 296 Bilder, im Median 320 Vögel pro Bild (5–95 %:
  142–666, max. 820), Speed-Median 6 m/s, Ausdehnung ≈ 100 × 100 × 40 m.

## Auswertung (`scripts/analyze_swifts.py`, Funktionen aus `scripts/analyze_real_birds.py`)

- **Je Bild:** genau wie O3b. Hauptschwarm (Einzelverknüpfung, 5 × medianer Nachbarabstand), mindestens 6 Vögel.
  c_nn unter **Q, L, RL, A7, M**, Bias-Korrektur je Bild (20 Wiederholungen, 3D).
- **Geschwindigkeit:** die der Autoren (vx, vy, vz).
- **Bildauswahl:** jedes 3. Bild (≈ 10 Hz wie die Simulation).
- **Einheit:** 10-s-Fenster, gültig mit mindestens 80 % gültigen Bildern.
- **Primär:** Q (N ≫ 30), dazu immer L.
- **Lesart** (Median über Fenster):

  | Lesart | Bedingung |
  |---|---|
  | schwach gekoppelt (wie Tauben) | Q < 0,05 und L < 0,10 |
  | stark gekoppelt (wie Dohlen) | Q ≥ 0,20 oder L ≥ 0,35 |
  | dazwischen (wie echte Drohnen) | sonst |

- **Beschreibend:**
  - AUC′ = P(T_Drohne > T_Segler) gegen die echten Drohnen (3D, q = 5) unter Q und L, über Fenster. Es gibt nur
    *eine* Aufnahme, also keinen Cluster-Bootstrap über Flüge, nur einen Punktwert.
  - Stärke der Drehung (|Ω| aus A7), weil der Schwarm kreist: Wie weit liegen A7 und M über Q/L?
- **Sensitivitäten:** jedes 6. Bild statt jedes 3.; 2-s-Fenster statt 10 s.
- **Grenzen, vorab benannt:** Kreisen vor dem Schlafplatz ist ein Sonderverhalten, es ist nur eine Aufnahme, und der
  Kameraabstand ist größer als bei den Dohlen (Messfehler unbekannt).

## Aufwand

Wenige Minuten Rechnung. Abweichungen werden im Bericht genannt.
