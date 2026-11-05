# Vorregistrierung: Bestätigungslauf und Rausch-Sensitivität

Festgeschrieben am 2026-10-11, **vor** dem Start von `P1_confirm_v1` und den Sensitivitätsläufen.
Anlass ist der explorative Befund aus `P1_small_v1`: Drohnen zeigen eine *größere*
Nachbar-Speed-Korrelation als Vögel (Forschungsbericht, Abschnitt 6.3). Die Projektleitung hat
beide Läufe freigegeben.

## A. Bestätigungslauf (konfirmatorisch)

| Punkt | Festlegung |
|---|---|
| Hypothese H1′ | Drohnengruppen haben eine größere Nachbar-Speed-Korrelation T_nn (Abzug A7) als Vogelschwärme. |
| Schätzgröße | AUC′ = P(T_Drohne > T_Vogel), U-Statistik ohne Diagonale, 95-%-Intervall per Designpunkt-Bootstrap (2000 Resamples, Perzentil) |
| Daten | `configs/p1_confirm.yaml`: dieselben Modelle, Parameter und Einstellungen wie `P1_small_v1` (inkl. Drohnen-Δt 5 ms), **neuer Seed 20261013**; N = 20, 30, 50 mit je 300 Designpunkten |
| **Einziger konfirmatorischer Test** | **N = 50: H1′ gilt als bestätigt, wenn AUC′ ≥ 0,8 für alle drei Regler (D1, D2, D3) und jeweils die untere Intervallgrenze > 0,7.** Sonst gilt H1′ als nicht bestätigt. |
| Beschreibend | N = 20 und 30; Auswertung nur mit zusammenhängenden Schwärmen; Schnitte nach Vogelmodell |
| Ausschlüsse | keine |

## B. Sensitivität des Drohnen-Prozessrauschens (beschreibend)

| Punkt | Festlegung |
|---|---|
| Variation | σ_d ∈ {0,1 (Basis), 0,3, 1,0} m/s^{3/2} |
| Design | dieselben Designpunkte und Zufallsströme wie A (gemeinsame Zufallszahlen); nur Drohnen neu gerechnet (`configs/p1_sens_sd03.yaml`, `p1_sens_sd10.yaml`); Vögel aus A |
| Berichtet | AUC′ je σ_d, N und Regler mit Intervall; Median von T_nn und Größe der Speed-Schwankung der Drohnen je σ_d |
| Erwartung (aus der Hypothese in Abschnitt 7 des Berichts) | AUC′ sinkt mit wachsendem σ_d, weil zusätzliches unabhängiges Rauschen die Nachbarkopplung der Drohnen verwässert |
| Lesart | Bleibt AUC′ bei σ_d = 1,0 und N = 50 für alle Regler ≥ 0,8, ist der Befund robust gegen diese Annahme. Fällt sie unter 0,7, hängt er wesentlich an der Rauschannahme. Dazwischen: teilweise robust. |
