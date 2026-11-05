# P3 Wind: Vorregistrierung

**Stand:** 2026-10-11, vor dem Hauptlauf. Dieses Dokument wird vor dem Start committet; Änderungen
danach werden als Abweichung mit Datum und Begründung im Ergebnisbericht vermerkt.

## 1. Ausgangslage und Ziel

P1 (idealer Sensor, kein Wind) hat einen stabilen, vorab in der Richtung festgelegten Befund
geliefert: **Drohnengruppen zeigen eine stärkere Korrelation der Geschwindigkeitsschwankungen
benachbarter Individuen (T_nn) als Vogelschwärme.** Bestätigt bei N = 50 mit neuen Zufallszahlen
(AUC′ = 0,85 / 0,89 / 0,99 für D1 / D2 / D3). Die Rausch-Sensitivität zeigte, dass der Effekt
schwächer wird, wenn die Drohnen mehr *unabhängiges* Rauschen haben.

Wind bringt eine andere Art Störung ins Spiel: räumlich glatte, **korrelierte** Böen, die beide
Klassen gleichermaßen treffen. P3 prüft, ob der Unterschied unter realistischem Herbstwind an
Flughäfen im Norddeutschen Tiefland erhalten bleibt (Architektur 3.2: „bei welchen
Windbedingungen AUC einbricht“), und wie viel davon auf eine unterschiedliche Windreaktion von
Vögeln und Drohnen zurückgeht (Leck-Zuordnung, Architektur 8.6).

## 2. Hypothese und Entscheidungsregel

**H-P3 (konfirmatorisch, einzige Testhypothese):** Bei N = 50 und realem Herbstwind (Kontrolle K0)
gilt für jeden Drohnenregler D1, D2, D3

> AUC′ = P(T_nn,Drohne > T_nn,Vogel) ≥ 0,8 **und** untere Grenze des 95-%-Intervalls > 0,7.

Urteil je Regler:

| Urteil | Bedingung |
|---|---|
| robust | AUC′ ≥ 0,8 und untere Grenze > 0,7 |
| abgeschwächt | AUC′ ≥ 0,7, aber nicht robust |
| eingebrochen | AUC′ < 0,7 |

Gesamturteil: **ROBUST**, wenn alle drei Regler robust sind; **EINGEBROCHEN**, wenn mindestens ein
Regler eingebrochen ist; sonst **ABGESCHWÄCHT**.

**Erwartung (keine Testhypothese):** Wind verringert AUC′ (ΔAUC_Wind > 0), umso mehr, je stärker
die Turbulenz in Flughöhe ist. Begründung: Turbulenz prägt beiden Klassen räumlich glatte,
korrelierte Schwankungen auf; dominieren diese, ähneln sich die Korrelationsstrukturen.

**Leck-Zuordnung (8.6, ohne Richtungshypothese):** ΔAUC_K-Wind = AUC′(K0) − AUC′(K-Wind).

## 3. Design

**Modelle** wie P1_small_v1 und P1_confirm_v1, unverändert:

- Vögel: B1/B3 aus dem Kalibrier-Posterior, Zusammenhalt über n_coh.
- Drohnen: D1, D2 (Reichweite lokal), D3; max_neighbors = 10, σ_d = 0,1.
- Sensor: idealer Sensor mit 10 Hz.
- Zeiten: T_burn = 30 s, T_obs = 10 s.

**Stichprobe:**

- Seed 20261014 (neu).
- N ∈ {20, 30, 50} mit je 300 Designpunkten, insgesamt 900.
- Vogelmodell je N stratifiziert.
- Abarbeitung mit N = 50 zuerst (konfirmatorisch), dann N = 30 und N = 20. Greift das Zeitlimit,
  fehlen zuerst Designpunkte mit N = 20. Indizes und Seeds hängen nicht von der Reihenfolge ab.

**Wind** (`configs/p3_wind.yaml`):

- **Wetterzustand je Designpunkt** (Kovariate, für alle Klassen gleich): ein zufällig gezogener realer
  10-Minuten-Zustand aus DWD-Messungen.
  - Stationen: Hamburg-Fuhlsbüttel, Bremen, Hannover und BER.
  - Zeitraum: September–November 2010–2025.
  - Nur Tageslicht einschließlich bürgerlicher Dämmerung (Sonnenhöhe > −6°): 414.927 Zustände.
  - Übernommen werden U10 und die Windrichtung.
  - Die Turbulenzintensität in 10 m kommt aus dem Böenfaktor: I_u = (FX_10/FF_10 − 1)/3,01 nach Wieringa (1973).
  - Bei U10 < 2 m/s oder fehlender Böe gilt der Median 0,141.
- **Profil:** U(z) = U10 · (z/10 m)^α mit α ~ U[0,1; 0,4] (Annahme aus 5.2).
- **Turbulenz:**
  - Spektrum nach von Kármán; Höhenabhängigkeit von σ und Längenskalen nach MIL-F-8785C.
  - Das Niveau kommt aus I_u10.
  - Synthese gitterfrei nach Kraichnan: 8 logarithmische Schalen pro Dekade, 2 Richtungen je Schale, kleinste Wirbel ℓ_min = 1 m.
  - Das Feld ist eingefroren und wird mit U(z0) advektiert.
  - Die Turbulenzrealisierung ist je Klasse unabhängig (env-Strom), über die Kontrollen gleich (CRN).
- **Windübertragung (4.1):** u = v + w − c·ŵ.
  - Vögel: c_B ~ U[0; 0,5], τ_B ~ U[2; 10] s.
  - Drohnen: c_D = 1, τ_D ~ U[1; 10] s.
  - Der Wind wird alle 5 ms ausgewertet und dazwischen gehalten.

**Kontrollen** (dieselben Designpunkte und Zufallszahlen):

| Kontrolle | Änderung |
|---|---|
| K0 | Standard: realer Herbstwind |
| K-Wind | Drohnen übernehmen die Windreaktion (c_B, τ_B) des Vogels desselben Designpunkts |
| K-Calm | derselbe Designpunkt ohne Wind (Vogel und Drohnen) |

**Variante „schnelle Böenregelung“** (`configs/p3_gustfast.yaml`, nur N = 50, gegen dieselben
Vögel):

- Die Drohnen schätzen den Wind mit τ_D ~ U[0,1; 0,5] s, also in der Größenordnung ihrer Geschwindigkeitsregelung.
- Sie regeln Böen damit weitgehend aus, wie es eine GPS-Geschwindigkeitsregelung mit Störgrößenunterdrückung tut.
- Die Variante ist eine Sensitivitätsanalyse und wird beschreibend berichtet.

## 4. Statistik

- **Statistik:**
  - Primär T_nn (Abzugsvariante A7).
  - AUC als Mann-Whitney-U-Statistik ohne Diagonale.
  - Designpunkt-Bootstrap mit 2000 Replikaten, Perzentilintervall 95 %.
- **Gepaarte Differenzen** (ΔAUC_Wind, ΔAUC_K-Wind, Variante − K0): gemeinsame Bootstrap-Gewichte für beide AUC-Werte.
- **Keine Ausschlüsse** im Primärtest. Sekundär werden berichtet:
  - nur zusammenhängende Schwärme;
  - Windklassen nach U10 (< 3, 3–6, ≥ 6 m/s) und Terzile von σ_u in Flughöhe;
  - Abzugsvariante A12, die zusätzlich lineare Scherung entfernt;
  - Läufe mit U10 > 10 m/s (Sturm-Stresstest, 5.4) markiert.
- **Auswerteskript:** `scripts/analyze_p3.py`, vor dem Lauf an Pilotdaten geprüft.

## 5. Abweichungen von der Architektur (bewusst, dokumentiert)

1. **Wetter:** DWD-Stationsdaten (Bootstrap realer Zustände) statt Weibull-Prior für U10; das ist eine
   leichte Form des optionalen Moduls 5.4 (dort ERA5). Entscheidung der Projektleitung:
   „echte historische Wetterdaten für September bis November“. Die Daten bestätigen den Prior:
   Die Weibull-Anpassung ergibt k = 2,08 und λ = 4,74 m/s gegenüber (2; 4,5).
2. **Turbulenzniveau** aus dem gemessenen Böenfaktor statt allein aus MIL-F-8785C. Die Höhenabhängigkeit,
   das Verhältnis σ_w/σ_u und die Längenskalen bleiben MIL.
3. **Kraichnan-Synthese mit geschichteten Schalen** statt Wellenzahlen aus der Dichte ∝ E(k). Die
   Kovarianz ist dieselbe, der Trägheitsbereich wird mit weniger Moden abgedeckt. Wirbel unter
   1 m entfallen, das ist etwa 1 % der Varianz.
4. **Anisotropie:** isotrope Synthese mit longitudinaler Integrallänge L_u, danach komponentenweise
   Reskalierung, wie in 5.3.3 Punkt 5 vorgesehen.
5. **Zeitdiskretisierung:** Der Wind wird höchstens alle 5 ms ausgewertet und gehalten. Ein Wert
   treibt Schätzer und Übertragung; 4.9 sieht zwei Auswertungen pro Schritt vor. Der Unterschied
   ist von der Ordnung Δt.
6. **K-Wind:** 8.6 lässt offen, welche Klasse angeglichen wird. Hier übernehmen die Drohnen die
   Vogelreaktion, analog zu K-Dichte.

## 6. Pilot (offengelegt)

- Umfang: 12 Designpunkte (4 je N) mit eigenem Seed 20261099, nicht Teil des Hauptlaufs.
- Zweck: nur Rechenzeit und Plausibilität (Zusammenhalt, Stationarität, Windgrößen).
- Die Pilotwerte von T_nn und AUC′ wurden gesehen. Bei n = 4 je N sind sie nicht aussagekräftig und gehen nicht in die Auswertung ein.

## 7. Rechenzeit (aus dem Pilot)

| Teil | CPU-Sekunden je Designpunkt |
|---|---|
| K0 | 4,1 |
| K-Wind | 2,5 |
| K-Calm | 1,7 |

Die effektive Parallelität mit 4 Prozessen liegt bei 3,4.

| Lauf | Umfang | Zeit | Zeitlimit (kein neuer Designpunkt danach) |
|---|---|---|---|
| Hauptlauf | 900 Designpunkte | ≈ 37 min | 50 min |
| Variante | N = 50, 300 Designpunkte | ≈ 5 min | 8 min |

Die Zeitlimits sichern das Budget von höchstens 1 h ab. Die Projektleitung hat Option A
(Hauptlauf mit allen Kontrollen plus Variante) am 2026-10-11 freigegeben.
