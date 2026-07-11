# mirror-sd -analyysi — opit oMLX:ään

Date: 2026-07-11
Analysoitu repo: [0xClandestine/mirror-sd](https://github.com/0xClandestine/mirror-sd)
Toteutussuunnitelma: [mirror_sd_adoption_plan.md](mirror_sd_adoption_plan.md)
Companion to: [dflash_mlx_integration.md](dflash_mlx_integration.md),
[dflash2_long_context_plan.md](dflash2_long_context_plan.md),
[5x_speedup_research.md](5x_speedup_research.md)

---

## 1. Mitä brancheja olemme toteuttaneet (tilannekuva)

| Branch | Sisältö | Tila |
|---|---|---|
| `feat/ngram-spec-decoding` | N-gram / prompt-lookup-spekulointi ilman draft-mallia; adaptiiviset gatet, nollakustannuksinen miss-polku | Toteutettu + benchmarkattu |
| `experiment/fused-tq-ngram-ab` | Fused-kernel A/B -kokeilu n-gram-spekuloinnin päälle | Kokeilu |
| `feat/cacheblend-kv-reuse` | CacheBlend: ei-prefix-chunkkien KV-uudelleenkäyttö (phases 0–3, default-off), content_hash-persistenssi | Toteutettu (oletuksena pois) |
| `feat/fused-int4-attention` | Fused compressed-domain verify-attention TurboQuant int4 KV:n päällä; query-chunked value dispatch | Toteutettu (v1), mergattu mainiin |
| `feat/sparse-prefill-draftfree` | MInference-tyylinen dynaaminen sparse prefill; maskless block-SDPA + fused Metal -kernel; admin-asetukset + stats | Stage 1–2 + E2E |
| `feat/mtp-multi-depth` (nykyinen) | MTPLX: monisyvyys-MTP-draftaus (`mtp_draft_depth`), auto-tune-spike, quantized-head guard + acceptance-floor auto-disable; lisäksi model optimization advisor P1/P2a | P1+P2 spike+P3 toteutettu |
| (main, aiemmin) | DFlash-MLX-integraatio (`DFlashEngine`), L1/L2 prefix cache, verify-modet (`dflash`/`adaptive`/`ddtree`) | Tuotannossa, ceiling `dflash_max_ctx` |

Yhteinen teema: **spekulatiivinen dekoodaus ja KV-cachen tehostaminen Apple Silicon
GPU:lla (MLX)**. Kaikki työmme ajaa sekä draftin että verifyn samalla GPU:lla —
tämä on tärkein ero mirror-sd:hen.

## 2. Mikä mirror-sd on

mirror-sd toteuttaa **DFlash block-diffusion -spekulatiivisen dekoodauksen niin, että
draft-malli ajetaan ANE:lla (Neural Engine) rinnakkain GPU:lla ajettavan
target-mallin kanssa**. Tulos: 85 tok/s Qwen3.5-27B-4bit M4 Maxilla, ja 90 %+
draft-ajasta piiloutuu verify-passin taakse.

Ydinkomponentit:

- **Heterogeeninen ajo:** GPU verifioi edellistä blokkia samalla kun ANE draftaa
  seuraavaa; token-vaihto zero-copy IOSurfacen kautta (unified memory).
- **W8A16-kvantisointi draftille:** int8-painot + per-channel fp16-skaalaus,
  ~2x kaistansäästö; draft-kernel-aika putosi 57 ms → 36 ms.
- **CoreML-kernelit Rustilla (PyO3/maturin):** draftin 5 kerrosta käännetään
  ANE-kerneleiksi, painot int8-vakioina graafissa.
- **Perustuu papereihin** "DFlash: Block Diffusion for Flash Speculative Decoding"
  (Ye et al. 2025) ja "Mirror Speculative Decoding" (Zhao et al. 2025).

Benchmark-havainto: draft-aika on tasainen (~191 ms) 64→4096 tokenin konteksteissa
→ pullonkaula on **FFN-painojen kaista**, ei attention-kompleksisuus.

## 3. Opit, joita voisimme ottaa käyttöön

### 3.1 Draft/verify-rinnakkaisuus (tärkein oppi) — sovellettavissa ilman ANE:akin

Meidän DFlash- ja MTP-polkumme ovat **sekventiaalisia**: draft → verify → draft…
mirror-sd:n keskeinen voitto ei ole ANE itsessään vaan **pipelinointi**: seuraavan
blokin draftaus alkaa heti kun edellisen verify on lähtenyt liikkeelle.
MLX:ssä tämän voi approksimoida myös GPU-only:na erillisillä streameillä
(`mx.new_stream` / async eval), tai CPU:lla ajettavalla kevyellä draftilla
(n-gram-polkumme on jo CPU-puolella — sen limitys verify-passin kanssa on halpa
kokeilu). **Ehdotus: spike `experiment/overlapped-draft`** — mittaa paljonko
draft-ajasta saadaan piiloon nykyisessä DFlash/MTP-loopissa.

### 3.2 ANE draft-mallille (isompi hanke, suuri potentiaali)

Draft-malli on pieni (5 kerrosta) ja muuttumaton → ideaali ANE-kandidaatti.
GPU vapautuu kokonaan targetille, ja unified memory tekee vaihdosta ilmaisen.
Kustannus on merkittävä: Rust/CoreML-toolchain, kernel-precision-sudenkuopat
(ks. 3.4), mallikohtainen käännös. **Ehdotus:** kirjataan `research_spikes_plan.md`:ään
spikeksi; mirror-sd on MIT-lisensoitu, joten sen `ane/src/dflash.rs`-kernelbuildereita
voi käyttää suoraan referenssinä tai vendoroida.

### 3.3 W8A16 draft-painoille

Me kvantisoimme KV-cachea (TurboQuant int4) mutta emme erikseen draft-mallin
*painoja* kaistaoptimoidusti. mirror-sd:n data osoittaa että pieni draft on
puhtaasti kaistarajoitteinen → int8-painot + fp16-scales antoi ~1.6x
draft-kernelissä. MLX tukee 8-bit-kvantisointia natiivisti — halpa kokeilu:
DFlash-draftin lataus 8-bittisenä vaikka target olisi 4-bit. Huomioi
`mtp_quantized_head_guard` -oppimme: kvantisoitu draft-head voi romahduttaa
acceptancen, joten sama acceptance-floor-automaattikytkin päälle.

### 3.4 Precision-sudenkuopat dokumentoitava tarkistuslistaksi

mirror-sd:n Phase 2 kaatui aluksi siihen, että per-head Q/K-normit laskettiin
globaalisti kaikkien kanavien yli (ei per-head 128 dim) ja RoPE-konventio
(neox vs. ei) ei täsmännyt MLX:ään. Meillä on ollut vastaavia fused-kernel-vaiheita
(`fused_int4_attention_plan.md`). **Ehdotus:** lisätään kernel-työn tarkistuslistaan:
per-head vs. globaali normi, RoPE-konventio, cosine-similarity-vertailu
referenssitoteutukseen ennen benchmarkkeja.

### 3.5 Benchmark-metodologia: draft-ajan tasaisuus diagnostiikkana

mirror-sd:n taulukko (throughput + α + draft-aika kontekstin funktiona) paljastaa
suoraan onko pullonkaula kaista vai attention. Meidän `perf_bench` -CLI:hin kannattaa
lisätä sama sarake-erittely: draft-ms, verify-ms, α per kontekstipiste — erityisen
hyödyllinen `dflash2_long_context_plan.md`:n Phase 1 -mittauksessa (Design A vs B).

### 3.6 Yhteys long-context-suunnitelmaamme

mirror-sd:n havainto "attention ei ole pullonkaula edes 4096 kontekstissa, FFN-kaista
on" tukee dflash2-suunnitelman **Design B:tä** (kvantisoitu full-context verify)
lyhyillä konteksteilla — mutta ei mittaa >4K:ta, joten Design A/B -päätös vaatii
silti oman Phase 1 -mittauksen ≥32K:ssa.

## 4. Priorisoitu suositus

1. **Halpa & heti:** benchmark-erittely (3.5) + precision-tarkistuslista (3.4).
2. **Spike, viikko:** overlapped draft/verify GPU-streameilla tai CPU-draftilla (3.1).
3. **Halpa kokeilu:** 8-bit draft-painot DFlash/MTP-draftille (3.3).
4. **Iso hanke, vain jos 2 osoittaa ettei GPU-only riitä:** ANE-draft
   mirror-sd:n Rust-kernelit referenssinä (3.2).

Kohta 2 on paras hyöty/kustannus: se on arkkitehtuurinen oppi joka hyödyttää
kaikkia spekulointipolkujamme (DFlash, MTP, n-gram) ilman uutta toolchainia.
