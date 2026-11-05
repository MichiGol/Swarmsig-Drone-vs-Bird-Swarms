# O3a „Echte Vogelschwärme gegen echte Drohnen“: Vorregistrierung

**Stand:** 2026-10-12, vor der Auswertung. Das Dokument wird vor dem Start committet.

## 1. Ziel

Der Literaturabgleich ([`O3_O4_Empirischer_Abgleich.md`](O3_O4_Empirischer_Abgleich.md)) legt nahe, dass unsere
Vogelmodelle zu schwach nachbarkorreliert sind. Echte Stare zeigen aber nur Werte unter Mittelwertabzug (M), und
deren Rohdaten sind nicht öffentlich. Jetzt liegen freie Rohdaten echter Schwärme vor. Damit messen wir T_nn
direkt, mit demselben Code wie bei den echten Drohnen (O4).

**Fragen:**

1. Wie stark sind echte Vögel nachbarkorreliert, unter dem Pflichtabzug A7 und unter M?
2. Sind echte Drohnen stärker gekoppelt als echte Vögel? Am saubersten ist der Vergleich Tauben gegen Drohnen,
   denn beide wurden mit GPS an jedem Individuum und 5 Hz gemessen.
3. Treffen unsere Vogelmodelle die echten Werte?

## 2. Daten

| Datensatz | Quelle | Inhalt |
|---|---|---|
| Dohlen | Ling et al. 2019, Proc. R. Soc. B; Dryad doi:10.5061/dryad.kb8js06, Spiegel Zenodo 5007024, CC0; `DataS1.txt` (md5 8290c914…) | 6 Transitschwärme (Flug zum Schlafplatz), Stereokameras, 60 Bilder/s, Positionen und Geschwindigkeiten in 3D; laut Artikel ≈ 4 cm Positionsfehler |
| Tauben | Papadopoulou et al. 2022 (Daten aus Sankey et al. 2021); Dryad doi:10.5061/dryad.905qfttkc, Spiegel Zenodo 6027115, CC0; `all_emp_data.csv` (md5 abaa894e…) | 43 Schwarmflüge von Brieftauben, GPS mit 5 Hz, nur horizontale Positionen (Länge/Breite, 6 Nachkommastellen); 16 Kontrollflüge ohne Falke (12 klein mit 7–10 Tauben, 4 groß mit 30–34), 27 Flüge mit Roboterfalke |
| Drohnen | Vásárhelyi et al. 2018, wie in O4 | 4 Flüge mit je ≈ 30 Drohnen, GPS mit 5 Hz, 3D |

**Vor der Vorregistrierung angesehen** (nur Aufbau, keine Korrelationsgröße):

- Spalten, Bildraten und Flugdauern;
- Schwarmgrößen, Vögel pro Bild und Bahnlängen (Dohlen: im Median 14–52 Vögel pro Bild, Bahnen im Median 1,6–4,4 s);
- Median von Speed und Nachbarabstand (Tauben ≈ 16–25 m/s und ≈ 1 m);
- Anteil fehlender Positionen (Tauben 16 %).

## 3. Aufbereitung

**Gemeinsame Regeln:**

- **Je Bild nur der Hauptschwarm:** größte Gruppe, in der Individuen verbunden sind, wenn ihr Abstand höchstens das
  5-Fache des medianen Nachbarabstands dieses Bildes beträgt (Einzelverknüpfung). Das entfernt Nachzügler und abgespaltene Teile.
- **Mindestens 6 Individuen pro Bild,** sonst fällt das Bild weg.
- **Kenngrößen je Bild** mit dem Simulationscode (`IdealStatistics`, Abzüge A7, M, A12). Innerhalb einer Einheit
  (Fenster oder Schwarm) wird gemittelt wie in der Simulation.

**Tauben:**

- Länge/Breite werden je Flug in lokale Meter umgerechnet (gleichabständige Näherung wie in O4), also in 2D.
- Die Geschwindigkeit stammt aus zentralen Differenzen über ±q Abtastpunkte. **Primär ist q = 5 (±1 s)**,
  Sensitivität q = 1 (±0,2 s). Verwendet werden nur Abtastpunkte, an denen beide Stützpositionen vorliegen.
- **Flugphase:** Die Schwarm-Mittelgeschwindigkeit liegt bei mindestens 10 m/s, das schließt Start und Landung aus.
- **Einheit:** 10-s-Fenster, schrittweise versetzt wie in O4. Ein Fenster zählt, wenn mindestens 80 % seiner Bilder gültig sind.
- **Primär:** Kontrollflüge. Sensitivität: Falkenflüge, nur Bilder mit mindestens 50 m Abstand des Schwarmzentrums zum Falken.

**Dohlen:**

- 3D. **Primär sind die Geschwindigkeiten der Autoren** (Spalten u, v, w).
- Sensitivität: zentrale Differenzen der Positionen über ±6 Bilder (±0,1 s) und ±15 Bilder (±0,25 s).
- **Einheit:** der ganze Schwarm (6 Werte), gemittelt über alle gültigen Bilder. Für Streuungsbilder zusätzlich 2-s-Fenster.

**Drohnen:**

- Aufbereitung wie in O4: Zeitkorrektur, Flugphase, 5-Hz-Raster, 10-s-Fenster, q = 5 primär und q = 1 als Sensitivität.
- Neu ist der Hauptschwarm-Filter von oben.
- **Für den Vergleich mit den Tauben** werden die Drohnen zusätzlich nur horizontal ausgewertet (2D). Das ist der
  primäre Vergleich, 3D ist Sensitivität.

## 4. Kenngrößen

- **Primär:** T_nn unter A7.
- **Sekundär:** T_nn und c_nn unter M (Definition der Starenstudien); A12 nur beschreibend.
- **Bias-Korrektur je Bild statt Tabelle.**
  - Die bisherige Tabelle deckt nur N = 10–100 in 3D ab. Die Tauben sind aber 2D, die Dohlenschwärme teils größer als 100.
  - Neu: Je Bild wird E[c_nn] unter unabhängigen Schwankungen auf *denselben Positionen* geschätzt:
    20 Wiederholungen, isotrope Gaußschwankungen mit σ = 5 % der mittleren Speed in der Dimension der Daten, gleicher Abzug.
  - T_nn = Mittel(c_nn) − Mittel(Bias).
  - **Prüfung:** Auf zufälligen Positionen wie beim Tabellenbau muss diese Korrektur die Tabelle (3D, N = 10–100)
    auf ±0,01 treffen. Wo beides geht (3D, 10 ≤ N ≤ 100), wird T_nn zusätzlich mit der Tabelle berichtet.

## 5. Vergleiche und Lesarten

**(a) Tauben gegen echte Drohnen** (primär: 2D, q = 5, A7, Kontrollflüge)

- AUC′ = P(T_Drohne > T_Taube) über 10-s-Fenster.
- 95-%-Intervall per Bootstrap über Flüge (Cluster), 2000 Wiederholungen. Getrennt berichtet für kleine (7–10) und große (30–34) Taubenschwärme, primär für alle zusammen.

| Lesart | Bedingung |
|---|---|
| Drohnen stärker gekoppelt (Richtung des P1-Befunds hält empirisch) | AUC′ ≥ 0,8 und Intervall über 0,5 |
| schwache Tendenz zu Drohnen | 0,5 < AUC′ < 0,8 und Intervall über 0,5 |
| kein belastbarer Unterschied | Intervall enthält 0,5 |
| Tauben stärker gekoppelt (H1-Richtung) | Intervall unter 0,5 |

**(b) Realismus der Vogelmodelle** (A7, je Datensatz)

- Der Median der echten Einheiten wird mit den simulierten Vögeln B1/B3 verglichen. Grundlage ist der P3-Hauptlauf mit
  Wind (K0) bei passendem N: kleine Tauben → N = 20 (kleinster simulierter Wert); große Tauben → N = 30; Dohlen → N = 50.
  Die Simulation nutzt die Tabellenkorrektur und einen idealen Sensor.

| Lesart | Bedingung |
|---|---|
| Vogelmodelle realistisch | echter Median innerhalb des 10–90-%-Bereichs der Simulation |
| Vogelmodelle zu schwach gekoppelt | echter Median über dem 90-%-Punkt |
| Vogelmodelle zu stark gekoppelt | echter Median unter dem 10-%-Punkt |

- Messrauschen senkt T_nn bei echten Daten. Liegt ein echter Wert *trotz* Rauschen über der Simulation, ist der Befund
  „zu schwach“ belastbar. Liegt er darunter, könnte das am Rauschen liegen. Die Sensitivitäten mit anderem
  Geschwindigkeitsfenster zeigen, wie stark.

**(c) Dohlen gegen Drohnen:** nur beschreibend, weil die Sensoren verschieden sind (Stereokamera gegen GPS).

## 6. Ausgaben und Aufwand

- **Skripte und Ausgaben:**
  - Daten: `scripts/bird_data.py` (Einlesen nur als Text bzw. CSV, kein mitgelieferter Code).
  - Auswertung: `scripts/analyze_real_birds.py`.
  - Tabellen: `reports/O3a_echte_Voegel_results.md` und CSVs.
  - Abbildung: `figures/fig17_echte_voegel_tnn.png`.
  - Bericht: `reports/O3a_Echte_Voegel_Bericht.md`.
- **Rechenzeit:** wenige Minuten, keine Simulation.
- **Abweichungen** von diesem Plan werden im Bericht einzeln genannt und begründet.
