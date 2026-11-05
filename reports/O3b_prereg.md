# O3b „Robuste Kenngröße“: Vorregistrierung

**Stand:** 2026-10-12. Das Dokument wird vor der Test-Suite und vor der Auswertung der echten Daten committet.

## 1. Ziel

In O3a zeigte sich ([`O3a_Echte_Voegel_Bericht.md`](O3a_Echte_Voegel_Bericht.md), Abschnitt 4):

- **A7** erzeugt bei *nicht starrem* Kurvenflug eine scheinbare Nachbarkorrelation.
- **M** erzeugt sie bei *starrer* Drehung.

Gesucht ist eine Kenngröße, die gegen beides robust ist und echte, lokale Speed-Kopplung weitgehend behält.
Mit ihr werden die echten Daten (Tauben, Dohlen, Drohnen) neu ausgewertet.

## 2. Kandidaten (`src/swarmsig/observables/modes.py`, getestet in `tests/test_modes.py`)

| Kürzel | Speed-Schwankung δs_i | entfernt |
|---|---|---|
| L | \|u_i\| − (a + b·r_i), kleinste Quadrate | Mittelwert und lineares Speed-Gefälle über den Schwarm |
| Q | \|u_i\| − Polynom 2. Grades in r_i | zusätzlich gekrümmte Speed-Felder (10 Parameter in 3D, 6 in 2D) |
| RL | \|u_i\| − \|U + Ω × r_i + D r_i\| (starre A7-Bewegung), danach wie L | Speed-Muster starrer Bewegung exakt, Rest linear |
| A7, M | wie bisher | Referenz, nicht wählbar |

Alle drei nutzen nur Speed-*Beträge*. Ein reiner Kurswechsel bei gleicher Speed ändert sie daher nicht.

## 3. Test-Suite (`scripts/o3b_statistic_suite.py`)

**Aufbau der künstlichen Schwärme:**

- Positionen wie beim Bau der Bias-Tabelle (Ellipsoid, Abstand 1), Speed V = 10.
- In jedem Szenario unabhängige Schwankungen: Speed 2 % von V, Kurs 1°.
- Die Stärken sind relativ zum Gyrationsradius R_g angegeben.
- Je Szenario 300 Schwärme. Bias-Korrektur je Bild wie bei echten Daten (5 Wiederholungen).

**Störungsszenarien (dürfen keine Kopplung vortäuschen):**

| Gruppe | Szenarien |
|---|---|
| 3D, N = 30 | unabhängig; nicht starrer Kurvenflug mit 10°, 30°, 60°, 90° Kursunterschied über ±R_g; starre Gierdrehung mit Ω R_g / V = 0,05, 0,15, 0,3, 0,5; Nicken 0,15; Dilatation D R_g / V = 0,05, 0,15; Drehung 0,15 + Kurvenflug 30° |
| 2D, N = 30 | unabhängig; Kurvenflug 30°, 60°; Drehung 0,15, 0,3; Dilatation 0,15 |
| 2D, N = 10 | unabhängig; Kurvenflug 30°; Drehung 0,15 |

**Signalszenarien (echte lokale Speed-Kopplung, Gauß-Feld mit Amplitude 5 % von V, Matérn ν = 0,5):**

| Gruppe | Szenarien |
|---|---|
| 3D, N = 30 | ℓ = 0,5 und ℓ = 1 (≈ 1–2 Nachbarabstände); ℓ = 1 mit Kurvenflug 30°; ℓ = 1 mit Drehung 0,15 |
| 2D, N = 30 | ℓ = 0,5 und ℓ = 1; ℓ = 1 mit Kurvenflug 30° |
| 2D, N = 10 | ℓ = 1 |

Nur berichtet werden die schwarmweiten Kopplungen ℓ = 2 und ℓ = 5. Sie sind von einem Speed-Gefälle kaum zu
unterscheiden, deshalb gehen sie nicht in die Regel ein.

## 4. Auswahlregel

1. **Robust** ist ein Kandidat, wenn er zwei Bedingungen erfüllt:
   - In *allen* Störungsszenarien gilt |T| ≤ 0,05.
   - In *allen* Signalszenarien ist der Erhalt ≥ 0,5. Erhalt bedeutet T(Kandidat) / T_M(gleiches Kopplungsfeld ohne Störung).
2. **Primäre Kenngröße** ist der robuste Kandidat mit dem höchsten mittleren Erhalt. Bei Gleichstand auf drei
   Nachkommastellen gilt die Reihenfolge L, RL, Q (weniger Annahmen zuerst).
3. **Ist kein Kandidat robust,** wird der mit dem kleinsten größten Störungswert gewählt. Er heißt dann
   „eingeschränkt robust“, und alle Ergebnisse an echten Daten gelten nur unter Vorbehalt.

**Vorwissen:** Den Kandidaten L habe ich in O3a an 7 Szenarien angesehen. Er war robust gegen Kurvenflug und
mäßige Drehung, versagte aber bei starker starrer Drehung. Q und RL sind ungetestet. Die Regel ist mechanisch und
wird nicht nachträglich geändert.

## 5. Auswertung der echten Daten (`scripts/analyze_real_birds.py`)

- **Ablauf:** Pipeline unverändert wie in O3a, also gleiche Einheiten, Filter, Bias-Korrektur je Bild, q und 2D/3D.
- **Abzüge:** gewählte Kenngröße S\* (primär), dazu A7 und M als Referenz.
- **(a) Tauben gegen Drohnen:**
  - Primär 2D, q = 5, Kontrollflüge, AUC′ = P(T_Drohne > T_Taube) unter S\*.
  - Cluster-Bootstrap über Flüge, 2000 Wiederholungen. Lesarten wie in O3a.
  - Sensitivitäten wie in O3a: kleine/große Schwärme, q = 1, Drohnen 3D, Falkenflüge.
- **(c) beschreibend:**
  - Dohlen gegen Drohnen.
  - Werte je Drohnenflug. Unter einer robusten Kenngröße sollte der Unterschied zwischen Diagonal- und Kreisflug
    kleiner werden als unter A7.
- **(b) Vergleich mit den Vogelmodellen** folgt in O3c. Die Simulation muss dafür mit S\* neu gerechnet werden,
  weil die Flugbahnen nicht gespeichert sind.
- **Erwartung, ohne Richtungstest:**
  - Ist der A7-Wert der Drohnen durch Kurvenflug überhöht, liegen sie unter S\* tiefer.
  - Tauben und Dohlen sollten sich wenig ändern, weil A7 und M bei ihnen übereinstimmen.

## 6. Aufwand

Test-Suite ≈ 2 min, echte Daten ≈ 9 min. Keine Simulation.

Die Projektleitung hat O3b am 2026-10-12 freigegeben.
