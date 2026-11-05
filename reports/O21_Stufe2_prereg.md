# O21 Teil 2, Stufe 2 „Unterscheidungsmerkmale nach Entfernung und Wetter“: Vorregistrierung

**Stand:** 2026-10-16. Das Dokument wird vor der Rechnung committet. Grundlage: Stufe 1 ([Bericht](O21_Stufe1_Bericht.md)).

## Frage

Bis zu welcher Entfernung kann welches Merkmal eine kleine Drohne von einem Vogel unterscheiden, und wie hängt das vom
Wetter ab?

**Zentrale Systemgröße, die unklassifizierte Zone:** Entfernungen, in denen das Radar erkennt, aber kein Merkmal
unterscheidet. Vögel dort lösen Alarme aus, die niemand bewerten kann. Gemessen wird sie als Anteil
1 − R_Unterscheidung/R_Erkennung je Ziehung.

## Merkmale und Modelle (je Ziehung aus Verteilungen; Annahmen A, Literatur L)

| Merkmal | Reichweite | Parameter |
|---|---|---|
| **Radar-Mikro-Doppler** (Rotorblätter) | größte Entfernung mit SNR_Blatt ≥ SNR_mD; SNR_Blatt = SNR_Rumpf + Δ_b; nur Rauschen und Dämpfung (die Blattechos liegen weit außerhalb des Regen-Dopplers); Radar wie Stufe 1 | Δ_b gleichverteilt −20…−10 dB (A); SNR_mD gleichverteilt 10–15 dB (A) |
| **Kamera, sichtbar** (nur tags) | min(Pixelreichweite, Sichtreichweite); Pixel: R = s/(n·IFOV), IFOV = Pixelgröße/f; Sicht nach Koschmieder: R = V/3,912·ln(C₀/ε) | s = 0,35 m (Drohne); n gleichverteilt 6–12 Pixel (Wiedererkennen, Johnson, L); Pixel 3,45 µm; f log-gleich 100–600 mm (A); C₀ gleichverteilt 0,2–0,6; ε gleichverteilt 0,02–0,05 (L/A) |
| **Wärmebild** (Tag und Nacht) | wie Kamera, Pixel 12 µm, f log-gleich 50–300 mm; atmosphärisch mit V_IR = k·V | k gleichverteilt 1,5–4 bei V ≥ 1 km, 1–1,5 bei Nebel (V < 1 km) (A); C₀/ε gleichverteilt 5–50 (A) |
| **Schwarmstatistik** (Teil 1, Stereokamera, tags) | min(Stereo-Reichweite, Sichtreichweite der Kamera); Stereo: Z_max = √(σ_max·f_px·B/δd) | σ_max = 5 cm (primär, aus O2), Sensitivität 2 und 10 cm; Basis B gleichverteilt 5–30 m; δd gleichverteilt 0,2–1 Pixel; f_px wie die Kamera (A) |
| **Funk** (passiv) | log-gleich 0,5–5 km, wetterunabhängig; **nur für Drohnen, die senden** | A (Herstellerbereich) |

Die Radar-Erkennung kommt aus Stufe 1, X-Band mit Echo-Unterdrückung 10–30 dB (vorregistriert) und 30–50 dB (Sensitivität).

## Wetterszenarien (vorab festgelegt)

| Szenario | Regen [mm/h] | Sicht V | Tageslicht |
|---|---|---|---|
| klarer Tag | 0 | 20 km | ja |
| Dunst | 0 | 5 km | ja |
| mäßiger Regen | 4 | 4 km | ja |
| Starkregen | 25 | 1,5 km | ja |
| Nebel | 0 | 0,3 km | ja |
| klare Nacht | 0 | 20 km | nein |

## Auswertung

- **Je Szenario und Merkmal:** Median sowie 10- und 90-%-Punkt der Reichweite (2 000 Ziehungen, dieselben Ziehungen in
  allen Szenarien).
- **Beste Unterscheidung:** R_U = max(Mikro-Doppler, Kamera, Wärmebild). Funk wird getrennt berichtet, weil er nur
  sendende Drohnen erfasst. Die Schwarmstatistik ebenfalls, weil sie nach Teil 1 nur gegen schwach gekoppelte Vögel trennt.
- **Unklassifizierte Zone:** 1 − min(R_U, R_E)/R_E je Ziehung, als Median.
- **Erwartungen, ohne Test:**
  - Mikro-Doppler reicht weit, aber kürzer als die Erkennung.
  - Kameras fallen bei Nebel und nachts aus, Wärmebild nur bei dichtem Nebel.
  - Die Schwarmstatistik reicht nur einige hundert Meter.

## Grenzen

- Die Merkmalsschwellen sind Annahmen. Sie werden als Verteilungen gezogen, die Ergebnisse sind Bandbreiten.
- Ein vorhandenes Merkmal ist noch keine sichere Unterscheidung. Wie gut es trennt (Fehlerraten), steht hier nicht im
  Modell. Die Reichweiten sind obere Grenzen.
- Die Kopplung von Regen und Sicht ist über Szenarien vorgegeben, nicht berechnet (echte Kombinationen folgen in Stufe 3).

## Ausgaben

- Code: `src/swarmsig/system/optics.py` und `radar.micro_doppler_range` (mit Tests), `scripts/o21_stage2_cues.py`.
- Ergebnisse: `reports/O21_Stufe2_results.md`.
- Abbildung: `figures/fig22_merkmale_reichweite.png`.
- Bericht: `reports/O21_Stufe2_Bericht.md`.
