# O20 „Zeitliche Signatur der Nachbarkopplung“: Vorregistrierung

**Stand:** 2026-10-16. Das Dokument wird vor der Auswertung committet.

## 1. Frage und Hypothese

Bisher messen wir nur, *ob* Nachbarn gemeinsam schwanken. Dohlen und Segler sind dabei stärker gekoppelt als der echte
Drohnenschwarm (O3b, O19). Hier fragen wir, *wie zeitlich* die Kopplung ist.

**Hypothese H_T:**

- **Vögel:** Sie nehmen ihre Nachbarn vor allem nach vorn wahr. Speed-Änderungen sollten daher *von vorn nach hinten*
  laufen: Der Vordermann ändert sich zuerst, der Hintermann folgt mit Verzögerung.
- **Drohnen:** Der Vásárhelyi-Regler empfängt alle Nachbarn per Funk richtungsunabhängig. Es sollte keine Vorn-hinten-Asymmetrie geben.
- **Folge:** Ist die Asymmetrie bei Vögeln positiv und bei Drohnen null, trennt sie Drohnen auch von *stark* gekoppelten
  Vögeln. Das ist die Lücke der bisherigen Kenngröße.

## 2. Kenngrößen

**Speed-Schwankung je Bild:** δs_i(t) nach Abzug **L** (Speed minus lineares Gefälle; primär, weil für alle Gruppengrößen
nutzbar). Hauptschwarm wie in O3b, mindestens 6 Individuen. Sensitivitäten: Q (nur bei N ≥ 30 sinnvoll) und M.

**Paare:**

- Für jedes Individuum i und seinen nächsten Nachbarn j zum Zeitpunkt t wird bestimmt, wer *vorn* ist.
- Vorn ist, wer in Flugrichtung des Hinteren liegt, also (x_vorn − x_hinten)·û_hinten > 0. Die eigene Flugrichtung
  wird genommen, nicht die mittlere, weil kreisende Schwärme keine gemeinsame Richtung haben.

**Zeitversetzte Korrelation:**

    C(k) = ⟨ δs_vorn(t) · δs_hinten(t + kΔ) ⟩ / ⟨ δs² ⟩

mit k = −5 … 5 und Δ = 0,2 s (also bis ±1 s). Gemittelt wird über alle Paare und Zeitpunkte einer Einheit.

**Primäre Kenngröße, die Asymmetrie:**

    A = [ Σ_{k=1..5} C(k) − Σ_{k=1..5} C(−k) ] / Σ_{k=−5..5} |C(k)|

A > 0 heißt: Der Hintere folgt dem Vorderen.

**Warum A unverzerrt ist:** Der Bias durch den Modenabzug und die Glättung der Geschwindigkeiten sind in der Zeit
symmetrisch. Bei unabhängigen Schwankungen ist A daher im Erwartungswert 0. Eine Tabellenkorrektur ist nicht nötig.

**Sekundär:** D = Σ_{k=1..5} C(k) − Σ_{k=1..5} C(−k), nicht normiert und damit weniger verrauscht bei schwacher Kopplung.

**Prüfung an künstlichen Schwärmen** (vor der Vorregistrierung, je 150 Schwärme mit N = 30, Bias-Prüfung des Codes):

| Fall | A |
|---|---|
| unabhängiges weißes Rauschen | −0,01 ± 0,02 |
| zeitlich korreliertes unabhängiges Rauschen (wie Glättung) | +0,01 ± 0,03 |
| Welle von vorn nach hinten | +0,14 ± 0,02 |
| Welle von hinten nach vorn | negativ (−0,20 ± 0,07 bei 20 Schwärmen) |

Die Kenngröße hat also keinen erkennbaren Bias und erkennt die Laufrichtung.

**Beschreibend:**

- die Spitzenverzögerung k* = argmax C(k);
- die Autokorrelationszeit τ_ac der eigenen Schwankung (erste Verzögerung mit Autokorrelation < 1/e). Sie hängt stark
  an der Glättung je Datensatz und wird nur mit Vorbehalt berichtet.

## 3. Daten und Einheiten (gemeinsame Abtastung 5 Hz)

| Datensatz | Abtastung, Geschwindigkeit | Einheit |
|---|---|---|
| echte Drohnen (Vásárhelyi) | 5-Hz-Raster wie O4; zentrale Differenz ±1 Rasterpunkt (±0,2 s, möglichst wenig Glättung); 3D | 10-s-Fenster |
| Tauben (Kontrollflüge) | 5 Hz; zentrale Differenz ±1 (±0,2 s); 2D; Flugphase ≥ 10 m/s | 10-s-Fenster |
| Dohlen | jedes 12. Bild (5 Hz); Geschwindigkeiten der Autoren; Schwarm 3 ausgeschlossen (Duplikat) | ganzer Schwarm |
| Segler | jedes 6. Bild (≈ 5 Hz); Geschwindigkeiten der Autoren | 10-s-Fenster |

Gültig ist eine Einheit mit mindestens 200 Paar-Zeitpunkten bei k = 0.

## 4. Auswertung und Lesart

- **Je Datensatz:** Median von A mit 95-%-Intervall per Bootstrap über Cluster, also über Flüge (Drohnen, Tauben),
  Schwärme (Dohlen) und Fenster (Segler; nur eine Aufnahme, Intervall daher zu eng).
- **Trennung:** AUC′ = P(A_Vogel > A_Drohne) je Vogelart.

| Lesart | Bedingung |
|---|---|
| H_T gestützt | Vögel: A > 0 mit Intervall über 0; Drohnen: Intervall enthält 0 oder ist kleiner als bei den Vögeln |
| Asymmetrie trennt | AUC′ ≥ 0,8 gegen Dohlen und gegen Segler |
| teilweise | sonst, wenn mindestens eine stark gekoppelte Art (Dohlen oder Segler) A > 0 mit Intervall über 0 zeigt |
| nicht gestützt | sonst |

- **Grenzen, vorab benannt:**
  - Die Drohnendaten sind GPS mit 5 Hz, die Vogeldaten teils Stereokameras mit von den Autoren geglätteten
    Geschwindigkeiten. Die Unterschiede in der Glättung betreffen vor allem τ_ac, kaum das Vorzeichen von A.
  - Die Dohlenbahnen sind kurz (im Mittel 2–4 s), das ergibt wenige Paare mit großer Verzögerung.
  - Die Simulation speichert keine Flugbahnen. Ein Vergleich mit den Modellen folgt gegebenenfalls später.

## 5. Aufwand

Wenige Minuten Rechnung (`scripts/analyze_temporal.py`).
