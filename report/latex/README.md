# ACL 2023 LaTeX report

Camera-ready-style course report using the official ACL 2023 class
(`acl.sty` + `acl_natbib.bst`), A4 two-column, Times, author–year citations,
mandatory Limitations section.

## Compile

From this folder, with a TeX install that has `pdflatex` and `bibtex`
(MiKTeX or TeX Live):

```bat
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf`.

Figures are read from `../figures/` (the PNGs copied from the Kaggle run).
If a figure is missing, copy it from the repo root `figures/` folder into
`report/figures/` first.

Overleaf: upload this `latex/` folder plus `report/figures/*.png`, set
`main.tex` as the compiler root, compiler = pdfLaTeX.
