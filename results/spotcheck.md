# Spot-check (minilm)

SciFact test query `1`:

> 0-dimensional biomaterials show inductive properties.

**Gold doc** `31715818` — *New opportunities: the use of nanotechnologies to manipulate and track stem cells.* (MiniLM rank **5**)

**MiniLM top-10** (2026-09-04, CPU):

1. `29638116` — Complex Tissue and Disease Modeling using hiPSCs.
2. `4346436` — Nonlinear Elasticity in Biological Gels
3. `3874000`
4. `10786948`
5. `31715818` — gold
6. `17388232`
7. `86129154`
8. `927561`
9. `19855358`
10. `1769799`

The first hits are thematically nearby (stem cells, biomaterials) but not the gold abstract. `bge-small` on the same query did **not** place `31715818` in the top-10 at all — a reminder that MRR 0.68 still misses individual claims.
