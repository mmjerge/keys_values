# ICML paper: RL through a bounded, evicting KV cache

## Getting it into Overleaf

Option A (one-off upload):

```bash
cd paper && zip -r ../icml_paper.zip . -x '*.pdf' '*.aux' '*.log'
```

Overleaf > New Project > Upload Project > pick `icml_paper.zip`. Set the
compiler to pdfLaTeX and the main document to `main.tex`.

Option B (keep in sync with this repo): Overleaf > New Project > Import from
GitHub, point it at this repo, and set `paper/main.tex` as the root document
in the project menu. Overleaf's GitHub sync is one project per repo, so if
that is awkward, push `paper/` to its own repo instead.

## Local build

```bash
cd paper && latexmk -pdf main.tex
```

## Layout

- `main.tex`: preamble, title block, section includes.
- `sections/`: one file per section. Red bracketed text (`\draft{...}`) marks
  material that still has to be written or filled with final numbers.
- `tables/`: tables that are generated from experiment logs. Each file names
  its source doc in a comment at the top.
- `figures/`: PDFs/PNGs for `\includegraphics`.
- `references.bib`: seed bibliography. Entries tagged `CHECK` were written
  from memory and need their venue confirmed.

## Style files

`icml2026.sty` / `icml2026.bst` are a stand-in. The ICML 2027 kit was not yet
posted as of 2026-09-30. When it appears at
`https://media.icml.cc/Conferences/ICML2027/Styles/`, drop the new `.sty`
and `.bst` here and update the two `icml2026` references in `main.tex`.
