# mirror-sd -oppien käyttöönotto — toteutussuunnitelma

Date: 2026-07-11
Suggested branches: `feat/specbench-breakdown` (P0), `experiment/overlapped-draft` (P1),
`experiment/draft-quant-defaults` (P2), `experiment/ane-draft-spike` (P3, ehdollinen)
Status: **planned**
Companion to: [mirror_sd_analysis.md](mirror_sd_analysis.md) (tausta ja perustelut),
[dflash_mlx_integration.md](dflash_mlx_integration.md),
[dflash2_long_context_plan.md](dflash2_long_context_plan.md),
[mtplx_adoption_plan.md](mtplx_adoption_plan.md)

Analyysin (mirror_sd_analysis.md §4) priorisointi: P0 halpa ja heti, P1 paras
hyöty/kustannus, P2 halpa kokeilu, P3 iso hanke vain jos P1 osoittaa ettei
GPU-only riitä.

---

## Tilannetarkennus analyysin jälkeen

Koodin tarkistus muutti yhtä kohtaa: **oppi 3.3 (W8A16 draft-painot) on jo
puoliksi toteutettu.** `DFlashEngine` tukee `draft_quant_enabled` /
`draft_quant_weight_bits` / `draft_quant_activation_bits` / `draft_quant_group_size`
-asetuksia ([dflash.py:138-153](omlx/engine/dflash.py#L138-L153)), ja ne on viety
`model_settings.py` / `model_profiles.py` / admin-reitteihin asti. P2 on siis
benchmark + oletusarvojen valinta, ei uutta toteutusta.

`scripts/perf_bench.py` ei erittele draft/verify-aikoja eikä acceptancea —
P0 on aito puute.

---

## P0 — Benchmark-erittely + precision-tarkistuslista

Branch: `feat/specbench-breakdown` · Työmäärä: ~1 päivä · Riski: matala

### P0.1 perf_bench: spekulointimetriikat kontekstin funktiona

mirror-sd:n taulukkomuoto (throughput, α, draft-ms per kontekstipiste) paljasti
suoraan että pullonkaula on FFN-kaista. Sama diagnostiikka meille:

- Lisää `scripts/perf_bench.py`:hin `--spec-breakdown`-lippu, joka tulostaa
  per kontekstipiste: `tok/s`, `draft_ms`, `verify_ms`, `α` (hyväksyttyjä
  tokeneita / verify-kutsu), `accept_rate`.
- Mittauslähde: DFlash- ja MTP-polut keräävät nämä jo osittain (MTPLX P3:n
  acceptance-floor tarvitsi acceptancen; DFlash-loopissa draft/verify erottuvat
  kutsurajalla). Puuttuvat ajastukset lisätään `time.perf_counter()`-pareina
  engine-looppiin **vain kun stats-keräys on päällä** — ei overheadia
  normaaliajossa.
- Vakiokontekstipisteet: 64, 512, 2048, 4096 (+ 8192 jos malli sallii), jotta
  tulokset ovat vertailukelpoisia mirror-sd:n taulukkoon.

**Hyväksymiskriteeri:** yhdellä komennolla saa mirror-sd-tyylisen taulukon
DFlash- ja MTP-moodeille; tulos liitetään dflash2-suunnitelman Phase 1
-mittaukseen (Design A vs B -päätös).

### P0.2 Kernel-precision-tarkistuslista

mirror-sd:n Phase 2 -sudenkuopat + omat fused-int4-oppimme yhteen paikkaan.
Lisätään tähän dokumenttiin (alla) ja linkitetään
[fused_int4_attention_plan.md](fused_int4_attention_plan.md):stä:

- [ ] Q/K-normit: per-head (head_dim) vai globaali? Tarkista molemmat toteutukset.
- [ ] RoPE-konventio: neox (half-split) vs. interleaved — täsmää MLX-referenssiin.
- [ ] Kvantisointiskaalat: per-channel vs. per-group, ja missä dtypessa skaalat.
- [ ] Cosine-similarity referenssitoteutukseen (>0.999 per kerros) **ennen**
      yhtään benchmarkia.
- [ ] Acceptance-vertailu (α) fp16-referenssiin ennen kvantisoidun polun hyväksyntää.

Ei koodia — dokumentaatiomuutos, mutta pakollinen portti P1–P3-työlle.

---

## P1 — Overlapped draft/verify -spike (tärkein)

Branch: `experiment/overlapped-draft` · Työmäärä: ~1 viikko · Riski: keskitaso

Nykyiset looppimme ovat sekventiaalisia: draft → verify → draft. mirror-sd:n
ydinvoitto on limitys. Tavoite: mittaa paljonko draft-ajasta saadaan piiloon
**ilman ANE:a**.

### Vaihe 1.1 — Mittaa limityspotentiaali (1 pv)

P0.1:n datalla: jos `draft_ms / (draft_ms + verify_ms)` < ~15 %, limitys ei
kannata → spike päättyy tähän ja P3 putoaa pois kokonaan. Kirjaa tulos tänne.

### Vaihe 1.2 — CPU-draft-limitys n-gram-polulle (helpoin voitto, 1–2 pv)

N-gram-draft (feat/ngram-spec-decoding) on jo CPU:lla → aidosti rinnakkainen
GPU:n kanssa ilman stream-kikkoja:

- Kun verify-forward on submittattu (`mx.async_eval`), laske seuraavan blokin
  n-gram-ehdokas CPU:lla ennen `mx.eval`-synkkaa.
- Spekulatiivinen draft: draftataan *oletetun* hyväksyntäprefiksin perään; jos
  verify hylkää aiemmin, heitetään draft pois (n-gram-draft on ~ilmainen, joten
  hukkatyö ei haittaa).

### Vaihe 1.3 — GPU-stream-limitys DFlash/MTP-draftille (2–3 pv)

- Kokeile `mx.new_stream(mx.gpu)` + `mx.stream(...)`-kontekstia: draft-forward
  omaan streamiin, verify oletusstreamiin, synkkaus vain token-vaihdossa.
- Realistinen odotus: GPU on jo saturoitunut verify-passin aikana, joten voitto
  voi jäädä pieneksi — **tämä nimenomaan on spiken kysymys**, ja negatiivinen
  tulos on P3:n go-peruste.
- Sama spekulatiivinen prefiksi-oletus kuin 1.2; rollback = draftin hylkäys.

### Vaihe 1.4 — Päätös ja raportti

Kirjaa tähän dokumenttiin: piilotettu osuus draft-ajasta (%), tok/s-delta,
α-muutos (spekulatiivinen prefiksi-oletus voi laskea α:aa). Go/no-go P3:lle:
jos limitys GPU-only:na piilottaa <30 % draft-ajasta *ja* draft-osuus
kokonaisajasta on >20 %, ANE-spike (P3) on perusteltu.

**Ei-tavoitteet:** ei muutoksia acceptance-semantiikkaan; default-off
(`spec_overlap_enabled`, oletus false) kunnes α-neutraalius on osoitettu.

---

## P2 — Draft-kvantisoinnin oletusarvot

Branch: `experiment/draft-quant-defaults` · Työmäärä: 1–2 pv · Riski: matala
Voi ajaa P1:n rinnalla; riippuu vain P0.1:stä.

Asetukset ovat jo olemassa — puuttuu data ja oletusarvot:

1. A/B-matriisi P0.1-työkalulla: draft fp16 vs. 8-bit vs. 4-bit
   (`draft_quant_weight_bits`), target vakiona. Mittarit: draft_ms, α, tok/s.
   mirror-sd:n perusteella odotus: 8-bit ≈ ~1.5x draft-kernel, α ennallaan;
   4-bit todennäköisesti syö α:aa.
2. Kytke MTPLX P3:n acceptance-floor-mekanismi (b1773b4) myös DFlash-draftin
   kvantisointiin: jos α putoaa alle floorin kvantisoidulla draftilla,
   pudota automaattisesti fp16-draftiin ja logita. Sama guard-periaate kuin
   `mtp_quantized_head_guard`.
3. Jos 8-bit on α-neutraali ja ≥1.2x draft-nopeus: aseta profiilioletukseksi
   `model_profiles.py`:hin ja päivitä model optimization advisor (P1/P2a,
   dfcfdee/f6f0b18) suosittelemaan sitä.

**Hyväksymiskriteeri:** mitattu taulukko tässä dokumentissa + oletusarvopäätös
+ guard toiminnassa (testi: keinotekoisesti huono draft → auto-fallback).

---

## P3 — ANE-draft-spike (ehdollinen: vain jos P1:n go-kriteeri täyttyy)

Branch: `experiment/ane-draft-spike` · Työmäärä: 2–4 viikkoa · Riski: korkea

Tavoite: DFlash-draftin 5 kerrosta CoreML/ANE-kerneleinä, GPU kokonaan
targetille. mirror-sd (MIT) toimii referenssinä — `ane/src/dflash.rs`:n
kernel-builderit (W8A16-dekvantisointi, per-head QK-norm, neox-RoPE) voi
vendoroida tai portata.

Vaiheistus (vasta go-päätöksen jälkeen tarkennetaan):

1. **Toolchain-spike (3 pv):** Rust + maturin + PyO3 buildautuu repossa;
   yksi dummy-ANE-kernel ajettavissa Pythonista. Ei mitään mallilogiikkaa.
2. **Draft-forward ANE:lla (1–2 vk):** mirror-sd:n kernelit + oma draft-bundle;
   P0.2-tarkistuslista porttina (cosine >0.999 vs. MLX-draft).
3. **Integraatio DFlashEngineen (3–5 pv):** ANE-draft `draft_backend`-vaihtoehtona
   (`EagerDraftBackend`-rinnalle, [dflash.py:308](omlx/engine/dflash.py#L308)),
   P1:n limityslooppi päälle.
4. **Benchmark + päätös:** vertailu P1:n GPU-only-limitykseen; hyväksyntäraja
   ≥1.3x tok/s GPU-only-limitettyyn nähden, muuten hylätään (toolchain-taakka
   ei muuten maksa itseään).

Riskit: CoreML-kääntäjärajoitteet (mirror-sd dokumentoi omansa repossa),
mallikohtainen käännösaika, upstream-synkkauksen vaikeutuminen (uusi
Rust-toolchain — pidetään kokonaan `ane/`-alihakemistossa erillään
upstream-tiedostoista, CLAUDE.md-sääntöjen hengessä).

---

## Riippuvuudet ja järjestys

```
P0.1 (bench) ──┬──> P1 (overlap-spike) ──go?──> P3 (ANE)
               └──> P2 (draft-quant)
P0.2 (checklist) ──> portti P1.3/P2/P3-kernelityölle
```

P0 ja P2 ovat riskittömiä ja tuottavat arvoa vaikka P1/P3 ei etenisi:
benchmark-erittely palvelee suoraan myös dflash2-long-context-suunnitelman
Phase 1 -päätöstä.
