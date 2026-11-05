# O21 Teil 2, Stufe 4 „Vogel-Fehlalarme mit echten Zugdichten“: Vorregistrierung

**Stand:** 2026-10-16. Das Dokument wird vor dem Herunterladen der Vogeldaten und vor der Rechnung committet.
Grundlage: Stufen 1–3 ([Stufe 3](O21_Stufe3_Bericht.md)).

## Frage

Wie viele Vögel erkennt ein Drohnenabwehr-Radar an norddeutschen Flughäfen, ohne sagen zu können, ob sie Drohnen sind?
Wir nennen sie **unbewertbare Vogelspuren**. Wie hängt ihre Zahl von Jahreszeit, Tageszeit, Wetter und Sensorkombination
ab, und um wie viel senken die Unterscheidungsmerkmale sie?

**Geometrie:** Ein Vogel in der Entfernung r ist unbewertbar, wenn das Radar ihn erkennt (r ≤ R_V, Erkennungsreichweite
für den Vogel) und kein Merkmal klären kann, ob dort eine Drohne fliegt (r > R_U, Unterscheidungsreichweite aus Stufe 2/3).
Die unbewertbaren Vögel liegen also in einem Ring zwischen R_U und R_V. Ist R_V ≤ R_U, ist der Ring leer.

## Neuheitsprüfung (Websuche am 16.10.2026, nicht systematisch)

- **Bekannt:**
  - Vögel sind die wichtigste Fehlalarmquelle von Drohnenradaren.
  - Der Mikro-Doppler ist das wichtigste Unterscheidungsmerkmal (z. B. Rahman & Robertson 2018).
  - Wetterradar-Zugdaten dienen der Vogelschlag-Warnung in der Militärluftfahrt (van Gasteren et al. 2019) und
    Zugvorhersagen (BirdCast).
- **Nicht gefunden:** eine Abschätzung der Vogel-Alarmlast eines Drohnenabwehrsystems aus Wetterradar-Zugprofilen, mit
  Parameterunsicherheit und echtem Wetter. Diese Kombination ist vermutlich neu; die Suche ersetzt O1 nicht.

## Daten

| Größe | Quelle | Verwendung |
|---|---|---|
| Vogeldichte je 200-m-Höhenstufe (`dens`, Vögel/km³, Umrechnung mit 11 cm² je Vogel) und Grundgeschwindigkeit (`ff`) | Aloft, Quelle baltrad, monatliche VPTS-CSV (5-min-Profile) | Säulendichte im Höhenfenster, Zugstrom |
| Wetter (Regen, Sicht, Tageslicht) | Stufe 3 (DWD, 10 min) am gepaarten Flughafen | Reichweiten je Intervall |

- **Paare Wetterradar ↔ Flughafen:**
  - Boostedt (`deboo`) ↔ Hamburg-Fuhlsbüttel
  - Hannover (`dehnr`) ↔ Hannover
  - Prötzel (`depro`) ↔ Berlin Brandenburg
  - Bremen entfällt, weil Aloft dort kein Wetterradar führt.
- **Zeitraum:** 2023–2025, drei vollständige Jahre. Aloft hat 2022 nur fünf Monate. Datenmenge ≈ 0,8 GB.
- **Zeit:**
  - Die wahre Profilzeit steht im Namen der Quelldatei; die Spalte `datetime` ist auf 15 min abgerundet.
  - Stundenmittel aus allen Profilen der Stunde; Stunden mit weniger als 3 Profilen gelten als fehlend.
  - Jedes 10-min-Wetterintervall erhält das Stundenprofil seiner Stunde (UTC).
- **Höhenfenster** (über NN, wie in den Profilen): primär 0–600 m.
  - Die Stufe 0–200 m ist im Wetterradar immer Lücke, weil der Strahl so tief nicht reicht. Sie wird mit der Dichte von
    200–400 m gefüllt (Annahme).
  - Fehlt die Dichte in einer beobachteten Stufe des Fensters, gilt die Stunde als fehlend.
- **Fehlende Geschwindigkeit:** Fehlt `ff` in einer Stufe, wird der dichtegewichtete Mittelwert der übrigen Stufen des
  Fensters eingesetzt, sonst 12 m/s.
- **Umgang mit den Dateien:** Die CSV-Archive werden nur als Daten gelesen, nichts wird ausgeführt.
- **Plausibilitätsprüfung:** Die Säulendichte sollte bei Regen (≥ 1 mm/h am Flughafen) im Mittel kleiner sein als bei
  Trockenheit. Andernfalls wäre Regen als Vogel gezählt worden.

## Modell

- **Gleiche Ziehungen:** wie in Stufe 2/3 (2 000 Ziehungen, Seed 20261017), X-Band-Radar.
- **R_U wie in Stufe 3:**
  - Hauptkombination X+K+W: Mikro-Doppler, Kamera nur tags, Wärmebild.
  - Dazu X allein (nur Mikro-Doppler).
  - Dazu der Vergleich ohne Unterscheidung (R_U = 0).
- **R_V:** X-Band-Erkennungsreichweite eines kleinen Vogels.
  - Rückstreufläche aus Stufe 1 (log-gleichverteilt 3·10⁻⁴ bis 3·10⁻³ m², Median ≈ 10 cm², passend zu den 11 cm² von
    Aloft).
  - Mit Regendämpfung und Regenecho wie für die Drohne.
- **Je 10-min-Intervall und Ziehung:**
  - **gleichzeitig unbewertbare Vögel:** N = n · π · max(R_V² − R_U², 0), mit n = Säulendichte im Höhenfenster in
    Vögeln/km².
  - **nie bewertbare neue Spuren pro Stunde:** Φ = n · v · 2 · max(R_V − R_U, 0). Das sind die Vögel, die den
    Erkennungskreis durchqueren, ohne je näher als R_U zu kommen; v ist die Grundgeschwindigkeit.
- **Annahmen:**
  - Horizontal gleich verteilt (Breitfrontzug).
  - Das Radar deckt das Höhenfenster im ganzen Ring ab.
  - Erkennung als Stufe bei P_d = 0,9.

## Kennzahlen

| Nr. | Kennzahl |
|---|---|
| W1 | Zeitmittel von N und Φ je Ziehung, Median [10 %; 90 %] über die Ziehungen; je Kombination (ohne Unterscheidung, X, X+K+W) |
| W2 | Anteil der Ziehungen mit nicht leerem Ring (R_V > R_U) bei klarem Wetter, tags und nachts; Zeitmittel von N für diese Ziehungen |
| W3 | Zeitanteil mit N ≥ 1, ≥ 10, ≥ 100 (je Ziehung; Median und 90-%-Punkt über die Ziehungen) |
| W4 | Reduktion durch Unterscheidung: N(X+K+W) / N(ohne Unterscheidung), Zeitmittel je Ziehung |
| W5 | Aufschlüsselung nach Jahreszeit × Tag/Nacht und je Flughafen; Jahres- und Tagesgang (Monat × Stunde) |
| W6 | Wetterkopplung (beschreibend): Säulendichte bei Regen gegen Trockenheit; Anteil der Vogelstunden nachts und je Jahreszeit |

## Sensitivitäten

- **S1:** mittlere statt kleine Vögel.
  - Rückstreufläche log-gleichverteilt 10⁻³ bis 3·10⁻² m² aus Stufe 1.
  - Die Dichte wird so skaliert, dass die gemessene Reflektivität erhalten bleibt (n · 11 cm² / σ).
- **S2:** Stufe 0–200 m gleich 0 (Untergrenze, nur Beobachtetes).
- **S3:** Höhenfenster 0–1000 m.
- **S4:** Echo-Unterdrückung + 20 dB.

## Erwartungen (ohne Test)

- **Ohne Unterscheidung:** In starken Zugnächten erkennt das Radar Hunderte bis Tausende Vögel gleichzeitig.
- **Mit Mikro-Doppler:**
  - Kleine Vögel werden meist nicht so weit erkannt, wie Drohnen unterschieden werden (R_V < R_U). In den meisten
    Ziehungen gibt es daher kaum unbewertbare Kleinvögel.
  - In einer Minderheit der Ziehungen (empfindliches Radar, schwaches Blattecho) sind es viele.
- **Zeitliche Verteilung:** Die Last liegt in Zugnächten im Frühjahr und Herbst, bei Regen ist sie gering.

## Grenzen

- **Genauigkeit der Profile:** Wetterradar-Dichten stimmen nachts etwa auf einen Faktor 2–3. Tagsüber können Insekten
  enthalten sein.
- **Lokale Vögel fehlen:** Möwen, Krähen, Tauben und Gänse am Flughafen fliegen meist tief und sind kaum erfasst. Die
  Last ist tagsüber eine Untergrenze.
- **0–200 m nicht beobachtet:** Die Stufe ist extrapoliert. Zudem gelten Höhen über NN statt über Grund (Unterschied
  < 150 m).
- **Räumliche Übertragung:**
  - Die Profile gelten für 5–35 km um das Wetterradar.
  - Boostedt und Prötzel liegen etwa 40 km von den Flughäfen entfernt, Hannover nah am Flughafen.
- **Vereinfachtes Radar:**
  - Kein Tracking, keine Schwarmbildung (jeder Vogel eine Spur).
  - Rückstreuflächen im X-Band wie im C-Band angenommen.
- **Keine positive Vogelerkennung:** Merkmale wie der Flügelschlag oder das Flugverhalten sind nicht modelliert. Sie
  würden die Last weiter senken.
- **Modellgrenzen aus Stufe 1–3 gelten weiter.**

## Ausgaben

- **Code:**
  - Funktionen für Ring, Zugstrom und Säulendichte mit Tests (`src/swarmsig/system/birds.py`,
    `tests/test_system_birds.py`)
  - `scripts/download_aloft.py`, `scripts/prepare_aloft.py`, `scripts/o21_stage4_birds.py`
- **Daten** (nicht versioniert): `data/raw/aloft/baltrad/<Radar>/`, `data/processed/aloft_hourly.parquet`.
- **Ergebnisse:** `reports/O21_Stufe4_results.md`, `reports/O21_Stufe4_load.csv`.
- **Abbildungen:** `figures/fig25_vogellast_jahresgang.png` (Jahres- und Tagesgang),
  `figures/fig26_vogellast_reichweite.png` (Erkennungsreichweite gegen Vogellast je Ziehung).
- **Bericht:** `reports/O21_Stufe4_Bericht.md`.
- **Aufwand:** Download ≈ 0,8 GB (je nach Leitung 2–10 min), Rechnung unter 10 min.

## Nachtrag vor der Rechnung (16.10.2026): Datenprüfung und Abweichung

Bei der Aufbereitung fielen sehr niedrige Vogeldichten der deutschen Radare 2023–2025 auf. Ein Vergleich derselben Radare
und Monate über die Jahre zeigt einen Bruch. Die Werte sind nächtliche, über alle Höhen summierte Dichten in Vögeln/km²,
angegeben als Median / Maximum der 5-min-Profile:

| Radar | Monat | 2017 | 2018 | 2019 | 2021 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| Boostedt | Okt | 11,6 / 501 | 7,5 / 288 | – | – | 0,3 / 71 | 0,1 / 18 | 0,1 / 11 |
| Hannover | Okt | 14,7 / 261 | 27,7 / 287 | 1,8 / 80 | 3,4 / 181 | 0,9 / 108 | 0,9 / 70 | 0,3 / 34 |
| Prötzel | Okt | 10,1 / 143 | 37,1 / 395 | – | – | 0,5 / 146 | 0,2 / 13 | 0,2 / 9 |
| Hannover | Sep | 16,6 / 83 | 26,2 / 161 | – | – | 1,5 / 16 | 0,7 / 26 | 0,3 / 19 |
| Prötzel | Apr | 6,5 / 41 | 27,4 / 478 | – | – | 1,4 / 72 | 0,1 / 33 | 0,2 / 10 |

- **2016–2018 sind plausibel:** In Zugnächten liegen die Werte bei Dutzenden bis Hunderten Vögeln/km², wie in der
  Literatur.
- **Ab 2019 fallen die Werte** schrittweise um das 10- bis 100-Fache, 2024/25 bei allen drei Radaren.
- **Deutung:** So ein Rückgang des Vogelzugs ist nicht plausibel. Wahrscheinlich hat sich die Verarbeitung der
  Radardaten geändert, sodass Vogelechos weitgehend herausgefiltert werden. Eine Dokumentation dazu haben wir nicht
  gefunden.

**Abweichung, festgelegt vor jeder Rechnung der Stufe 4:**

1. **Hauptanalyse mit dem Jahr 2018.** Es ist das letzte vollständige Jahr mit plausiblen Profilen. Das Wetter dazu
   kommt aus den DWD-Archiven 2010–2019, mit denselben Stationen und derselben Aufbereitung wie in Stufe 3.
2. **Lücken 2018:** Januar und Juli sind fast leer, Februar und Juni etwa halb. Jahresmittel werden deshalb als Mittel
   der Monatsmittel gebildet, jeder Monat gleich gewichtet.
   - Monate mit weniger als 72 gültigen Stunden (3 Tagen) entfallen und werden genannt.
   - Dieselbe Regel gilt für die Variante 2023–2025.
3. **Die vorregistrierte Variante 2023–2025** wird ebenfalls gerechnet und als „Profile vermutlich gefiltert“ berichtet,
   also als Untergrenze.
4. **Alles andere bleibt** wie vorregistriert: Paare, Höhenfenster, Modell, Kennzahlen und Sensitivitäten.
