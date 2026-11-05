# O21 Teil 2, Stufe 1 „Erkennungsreichweite bei Regen“: Vorregistrierung

**Stand:** 2026-10-16. Das Dokument wird vor der Rechnung committet. Parameter und Quellen: [`O21_Stufe0_Parameter.md`](O21_Stufe0_Parameter.md).

## Fragen

1. Wie weit erkennt ein bauartneutrales Drohnenabwehr-Radar eine kleine Drohne bei klarer Luft, und wie stark schrumpft
   die Reichweite bei Regen?
2. Ab welcher Regenrate reicht ein S-Band-Radar weiter als ein X-Band-Radar gleicher Antennenfläche und Leistung?
3. Wie weit werden Vögel erkannt? Wie stark überlappen Drohnen und Vögel? Damit wird der Bedarf an Unterscheidungsmerkmalen
   beziffert.

## Szenarien

- **Raster:** Band {X, S} × Regenrate {0; 1; 4; 10; 25; 50} mm/h × Ziel {kleine Drohne, kleiner Vogel, mittlerer
  Vogel, großer Vogel}.
- **Wiederholungen:** Je Zelle 4 000 Monte-Carlo-Ziehungen der Radar-, Ziel- und Clutterparameter. **Dieselben Ziehungen**
  gelten für alle Regenraten und für beide Bänder (gemeinsame Zufallszahlen). Damit sind Vergleiche gepaart.
- **Reichweite:** die größte Entfernung, bei der das Signal-zu-Rausch-plus-Clutter-Verhältnis die Swerling-1-Schwelle
  (21,1 dB) erreicht. Bestimmt per Bisektion zwischen 10 m und 100 km. Das Verhältnis fällt mit der Entfernung monoton,
  weil Signal ∝ R⁻⁴·Dämpfung und Signal/Clutter ∝ R⁻².

## Auswertung

- Je Zelle: Median sowie 10- und 90-%-Punkt der Reichweite.
- **Q1:** gepaarter Reichweitenverlust R(Regen)/R(0) je Ziehung, als Median.
- **Q2:** gepaarter Vergleich S gegen X je Ziehung, also Anteil der Ziehungen mit R_S > R_X je Regenrate. Der Übergang
  ist die Regenrate, ab der der Anteil 50 % übersteigt.
- **Q3:** Reichweite je Zieltyp. Der Überlappungsanteil P(R_Vogel > R_Drohne) beschreibt, wie oft ein Vogel weiter
  erkannt wird als eine Drohne.

## Plausibilitätsprüfung (vorab festgelegt)

- **Ziel:** Bei klarer Luft muss der Median der X-Band-Reichweite für die kleine Drohne zwischen 1 und 10 km liegen.
  Das ist die Spanne der Literatur- und Herstellerangaben.
- **Falls nicht:** Die Radarparameter werden **nicht** stillschweigend angepasst. Die Abweichung wird berichtet, und eine
  begründete Korrektur folgt als eigener, gekennzeichneter Schritt.

## Grenzen (vorab benannt)

- kein Bodenclutter und keine Mehrwegeausbreitung, kein Gelände, Erdkrümmung vernachlässigt (bis einige km unkritisch);
- Regen gleichmäßig über den ganzen Pfad;
- keine Signalsimulation;
- die Rückstreuflächen im S-Band sind teils angenommen.

## Ausgaben

- Code: `src/swarmsig/system/radar.py` (mit Tests) und `scripts/o21_stage1_ranges.py`.
- Ergebnisse: `reports/O21_Stufe1_results.md` und CSV.
- Abbildung: `figures/fig21_radar_reichweite.png`.
- Bericht: `reports/O21_Stufe1_Bericht.md`.
