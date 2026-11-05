# O21 Teil 2, Stufe 3 „Echtes Flughafenwetter: Verfügbarkeit je Sensorkombination“: Vorregistrierung

**Stand:** 2026-10-16. Das Dokument wird vor dem Herunterladen der Daten und vor der Rechnung committet. Grundlage:
Stufe 1 ([Bericht](O21_Stufe1_Bericht.md)) und Stufe 2 ([Bericht](O21_Stufe2_Bericht.md)).

## Frage

An wie viel Prozent der Zeit erreicht welche Sensorkombination bei echtem Wetter an norddeutschen Flughäfen welche
Erkennungs- und Unterscheidungsreichweite für eine kleine Drohne?

Dazu eine zweite Frage, die Stufe 2 nicht beantworten konnte: **Fallen die Schwächen der Sensoren im echten Wetter
zusammen?** In Stufe 2 waren Regen, Sicht und Tageslicht als Szenarien vorgegeben. Ein hybrides System lohnt sich nur,
wenn die Sensoren nicht gleichzeitig ausfallen.

## Daten (DWD Climate Data Center, offene Daten)

| Größe | Produkt | Auflösung | Verwendung |
|---|---|---|---|
| Niederschlag | `10_minutes/precipitation/historical` (RWS_10 in mm je 10 min, Kennung RWS_IND_10) | 10 min | Regenrate R = 6 · RWS_10 in mm/h |
| Sichtweite | `hourly/visibility/historical` (V_VV in m; V_VV_I: Beobachter oder Instrument) | 1 h | jedem 10-min-Intervall die zeitlich nächste Stundenbeobachtung (± 30 min) |
| Wind | `10_minutes/wind/historical` (FF_10), aus P3 bereits vorhanden | 10 min | nur Kontext (V8) |
| Tageslicht | Sonnenhöhe an der Station (NOAA-Näherung, wie in P3) | 10 min | Tag: Sonnenhöhe > 0° |

- **Stationen:** 00427 Berlin Brandenburg, 00691 Bremen, 01975 Hamburg-Fuhlsbüttel, 02014 Hannover (wie P3).
- **Zeitraum:** 2020-01-01 bis 2025-12-31, UTC.
- **Gültige Intervalle:** mit Regen- und Sichtwert. Fehlende Werte werden ausgeschlossen. Die Abdeckung wird je Station
  berichtet, die Qualitätsniveaus (QN) ebenfalls; gefiltert wird nicht.
- **Umgang mit den Dateien:** Die ZIP-Archive werden nur als Text/CSV gelesen, nichts wird ausgeführt.

## Modell

- **Unverändert aus Stufe 2:** gleicher Code, gleiche 2 000 Ziehungen (Seed 20261017), kleine Drohne. Erkennung über
  das Rumpfecho (X-Band, Echo-Unterdrückung 10–30 dB), Mikro-Doppler, Kamera, Wärmebild, Schwarmstatistik, Funk.
- **Neu, S-Band:** wie Stufe 1 (3,0 GHz, gleiche Antenne und Leistung, Drohnen-RCS × Faktor 0,3–1).
  - Mikro-Doppler im S-Band mit einem Blattecho, das **5 dB schwächer** ist als im X-Band (Annahme A: dünne Leiter
    streuen etwa proportional zur Frequenz). Sensitivität 0 und −10 dB.
- **Zeitmittel je Ziehung:** Jede Kennzahl wird für jede Ziehung über alle gültigen Intervalle gemittelt (Zeitanteil).
  Berichtet wird Median [10 %; 90 %] über die Ziehungen. So bleiben Wetterschwankung (über die Zeit) und
  Parameterunsicherheit (über die Ziehungen) getrennt.
- **Technische Umsetzung:**
  - Regenraten werden auf 40 logarithmische Stufen von 0,06 bis 150 mm/h gerundet (nächste Stufe); trockene Intervalle
    bleiben exakt 0.
  - Sichtweiten sind diskrete Meldewerte und werden exakt verwendet.
  - Die Reichweiten werden je Stufe und Ziehung einmal berechnet und dann den Intervallen zugeordnet.

## Sensorkombinationen (vorab festgelegt)

| Kürzel | Erkennung | Unterscheidung R_U |
|---|---|---|
| X | X-Band | Mikro-Doppler X |
| S | S-Band | Mikro-Doppler S |
| X+K | X-Band | max(Mikro-Doppler X, Kamera) |
| **X+K+W** (Hauptkombination, wie R_U in Stufe 2) | X-Band | max(Mikro-Doppler X, Kamera, Wärmebild) |
| S+K+W | S-Band | max(Mikro-Doppler S, Kamera, Wärmebild) |
| X+S+K+W | max(X, S) | max(Mikro-Doppler X, Mikro-Doppler S, Kamera, Wärmebild) |

Funk (nur sendende Drohnen) und Schwarmstatistik (nur gegen schwach gekoppelte Vögel) werden wie in Stufe 2 getrennt
berichtet.

## Kennzahlen

| Nr. | Kennzahl | Kombinationen |
|---|---|---|
| V1 | **Erkennung:** Zeitanteil mit R_E ≥ 1, 2, 3, 5 km | X, S, X+S |
| V2 | **Unterscheidung:** Zeitanteil mit R_U ≥ 0,5, 1, 2, 3 km | alle |
| V3 | **Doppelte Bestätigung** durch zwei physikalisch verschiedene Prinzipien: Zeitanteil, in dem Mikro-Doppler *und* ein Bildsensor (Kamera oder Wärmebild) beide mindestens r reichen; r = 0,3, 0,5, 1, 2 km | X+K+W, X+S+K+W |
| V4 | **Unklassifizierte Zone** 1 − min(R_U, R_E)/R_E: Zeitmittel je Ziehung und Zeitanteil mit Zone ≥ 0,5 | X, X+K+W, X+S+K+W |
| V5 | **Wetterverlust ohne Parameterunsicherheit:** Zeitanteil, in dem eine Ziehung mindestens 80 % (bzw. 50 %) ihrer eigenen Reichweite bei klarem Tag erreicht, für R_E und R_U | X, S, X+K+W |
| V6 | **Beitrag der Bildsensoren:** Zeitanteil, in dem Kamera oder Wärmebild weiter reicht als der Mikro-Doppler; mittlerer Reichweitengewinn R_U(X+K+W) − R_U(X) | X+K+W |
| V7 | **Fallen die Schwächen zusammen?** V1 (X ≥ 3 km), V2 (X+K+W ≥ 1 und 2 km), V3 (≥ 0,5 und 1 km) und V4 mit echtem Wetter gegenüber unabhängigen Wetterelementen. Dafür wird die Sichtweite innerhalb Station × Kalendermonat zufällig neu zugeordnet (20 Permutationen); Regen, Tageslicht und Jahresgang bleiben erhalten | siehe links |
| V8 | **Wind und Schwarmstatistik:** Verteilung von U10 in den P3-Klassen (< 3, 3–6, ≥ 6 m/s), wenn die Schwarmstatistik nutzbar ist (Tag und V ≥ 1 km), gegenüber der gesamten Zeit | – |

- **V7, beschreibend dazu:** P(V < 1 km) gesamt, nachts und bei U10 < 3 m/s; P(R ≥ 1 mm/h) gesamt und bei U10 ≥ 6 m/s;
  P(V < 5 km | R ≥ 4 mm/h).
- **V8, Begründung:** Nach O5 sind Formationsdrohnen nur bei Wind gekoppelt. Eine Windschwelle hat O5 nicht bestimmt,
  V8 ist deshalb rein beschreibend.
- **Aufschlüsselung** je Station und Jahreszeit (Dez–Feb, Mär–Mai, Jun–Aug, Sep–Nov) für V1 (X ≥ 3 km),
  V2 (X+K+W ≥ 2 km), V3 (≥ 1 km) und V4 (X+K+W).
- **Wetterbeschreibung:**
  - Zeitanteile mit Niederschlag > 0, ≥ 1, ≥ 4, ≥ 10, ≥ 25 mm/h und mit Sicht < 0,3, < 1, < 5 km; Nachtanteil.
  - Anteil der Niederschlagszeit mit beheiztem Messgerät (laut DWD-Parameterbeschreibung, wird nach dem Download
    geprüft) als grober Hinweis auf Frost und Schnee.

## Kontrollrechnung

Bei Regen 0 müssen Erkennung und Mikro-Doppler im X-Band exakt die Stufe-2-Mediane für „klarer Tag“ ergeben
(7,17 und 4,96 km), weil dieselben Ziehungen verwendet werden.

## Sensitivitäten

- **S1:** Echo-Unterdrückung + 20 dB (30–50 dB).
- **S2:** Tageslicht schon ab Sonnenhöhe −6° (bürgerliche Dämmerung).
- **S3:** S-Band-Blattecho 0 und −10 dB statt −5 dB.

## Erwartungen (ohne Test)

- Niederschlag fällt nur in einem kleinen Teil der Zeit (Größenordnung 5–10 %), Starkregen sehr selten (< 1 %). Der
  Jahresverlust an Erkennung durch Regen ist daher klein gegenüber der Parameterunsicherheit.
- Die Kamera fällt vor allem nachts aus (≈ 50 % der Zeit). Nebel kostet nur wenige Prozent.
- Doppelte Bestätigung bis 1 km ist etwa so häufig wie „Tag mit guter Sicht“.
- Nebel tritt vor allem nachts und bei Windstille auf. Dann fehlen gleichzeitig die Kamera und die Windhilfe für die
  Schwarmstatistik.

## Grenzen

- **Punktmessung:** Regen und Sicht werden an der Station gemessen und als gleichförmig entlang des Radarpfads
  angenommen. Schauerzellen sind nur wenige Kilometer groß. Für Pfade unter 10 km mindert ITU-R P.530 den Pfadregen
  aber nur um höchstens etwa 20 %.
- **Schnee:** Alle Niederschläge werden als Regen behandelt, obwohl Schnee anders dämpft und streut. Der Anteil
  wird berichtet.
- **Höhe:** Sicht und Wind werden am Boden gemessen, Drohnen fliegen höher (flacher Bodennebel, Wolkenuntergrenze,
  stärkerer Wind in Flughöhe).
- **Glättung:** 10-min-Summen glätten Regenspitzen, die Stundensicht glättet Nebelbänke.
- **Modellgrenzen** aus Stufe 1 und 2 gelten weiter: kein Bodenclutter, optimistische Mikro-Doppler-Reichweite, ein
  vorhandenes Merkmal ist noch keine Trefferquote.

## Ausgaben

- **Code:**
  - `src/swarmsig/system/climate.py` (Sonnenhöhe, DWD-Leser; Tests in `tests/test_system_climate.py`)
  - `scripts/download_dwd_o21.py`, `scripts/prepare_dwd_o21.py`, `scripts/o21_stage3_climate.py`
- **Daten** (nicht versioniert): `data/raw/dwd/zip_rr/`, `data/raw/dwd/zip_vv/`, `data/processed/dwd_o21_weather.parquet`.
- **Ergebnisse:** `reports/O21_Stufe3_results.md`, `reports/O21_Stufe3_availability.csv`.
- **Abbildungen:** `figures/fig23_verfuegbarkeit.png`, `figures/fig24_tages_jahresgang.png`.
- **Bericht:** `reports/O21_Stufe3_Bericht.md`.
- **Aufwand:** Download etwa 30 MB, Rechnung unter 5 min.
