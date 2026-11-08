# Data request to the Federal Statistical Office (BFS), Sektion Gesundheit

Sent: 2026-11 (fill in date). Contact: gesundheit@bfs.admin.ch. Reply and any delivered
tables are kept under `data/raw/bfs_request/` (not committed) and described in
`docs/wiki/01_data_sources.md` once received. What is asked for, in one sentence: deaths by
cause × canton × sex × 5-year age × 5-year period, 1995–2024, for all cancers and the main
sites, as a plain table.

---

**Betreff:** Anfrage Todesursachenstatistik nach Kanton, bösartige Neubildungen 1995–2024

Sehr geehrte Damen und Herren

Ich arbeite an einem nicht-kommerziellen, offenen Forschungsprojekt zur Entwicklung der
Krebsinzidenz und -mortalität in der Schweiz auf Basis Ihrer öffentlichen Daten (PxWeb-Würfel
px-x-1403030300_161/162, Todesursachentabellen je-d-14.03.04.01.01–04, su-d-14.03.04.01.11–20
sowie STATPOP). Code, Methodik und Ergebnisse sind öffentlich:
https://github.com/maltsev-andrey/cancer-ch

Der nächste Schritt ist eine räumliche Analyse auf Kantonsebene (BYM2-Modell). Dafür
benötige ich die Todesursachenstatistik nach Kanton, die in PxWeb und im Tabellenkatalog
nach 1972 nicht publiziert ist. Ich bitte Sie um folgende Tabelle, wenn möglich als CSV
oder XLSX:

**Inhalt:** Anzahl Todesfälle (Wohnbevölkerung, Grundtodesursache)

**Gliederung:**
- Kanton (26)
- Geschlecht (2)
- Altersklasse: 5-Jahres-Gruppen 0–4 … 80–84, 85+ (18); falls aus Datenschutzgründen
  nötig, alternativ 0–39, 40–49, 50–59, 60–69, 70–79, 80+
- Periode: 1995–1999, 2000–2004, 2005–2009, 2010–2014, 2015–2019, 2020–2024
  (5-Jahres-Summen genügen; Einzeljahre nur, wenn ohne Mehraufwand möglich)
- Todesursache, ICD-10:
  - Bösartige Neubildungen gesamt (C00–C97)
  - Lunge, Bronchien, Luftröhre (C33–C34)
  - Dickdarm und Rektum (C18–C21)
  - Brust (C50)
  - Prostata (C61)
  - Bauchspeicheldrüse (C25)
  - Magen (C16)
  - Leber (C22)
  - Hautmelanom (C43)
  - Non-Hodgkin-Lymphom (C82–C86)
  - Leukämie (C91–C95)
  - Gebärmutterhals (C53)
  - Harnblase (C67)
  - Niere (C64)
  - Mundhöhle und Rachen (C00–C14)
  - Speiseröhre (C15)

Falls einzelne Zellen aus Datenschutzgründen unterdrückt werden müssen (kleine Kantone,
seltene Lokalisationen), ist eine Unterdrückung mit Kennzeichnung oder eine gröbere
Altersgliederung für diese Kantone für mich in Ordnung. Die Vollständigkeit der Gliederung
nach Kanton ist wichtiger als die Feinheit der Altersgruppen.

Die Daten würden unter Angabe der Quelle (BFS, Todesursachenstatistik) ausschliesslich für
diese Analyse verwendet; abgeleitete Ergebnisse (Modellparameter, Karten) würden öffentlich,
die Rohtabelle selbst nur, wenn Sie dem ausdrücklich zustimmen.

Bitte teilen Sie mir mit, falls für diese Anfrage Kosten anfallen oder ein formelles
Gesuch nötig ist.

Mit freundlichen Grüssen

Andrey Maltsev
(Adresse, Ort)
andrey.maltsev@yahoo.com

---

## Notes for when the data arrive

- Check the canton list against `config/` (26 cantons; Basel-Stadt and Basel-Landschaft
  separate; no "Schweiz" row mixed in).
- Check totals against the national series: sum over cantons per cause × sex × age × period
  must match `mort_age.parquet` (je-d-14.03.04.01.01/02) within rounding and suppression.
- Suppressed cells: record the suppression rule in `docs/wiki/01_data_sources.md`; the BYM2
  likelihood can treat a suppressed cell as censored (count ≤ threshold) rather than missing.
- Denominators: mid-year population by canton × sex × 5-year age from `population_5y.parquet`
  (s04 already builds it from px-x-0102020000_104), nothing new to download.
