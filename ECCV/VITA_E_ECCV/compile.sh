latexmk -pdf -interaction=nonstopmode -file-line-error eccv2026_conference.tex
latexmk -pdf '-interaction=nonstopmode' -file-line-error '-jobname=appendix' .codex_appendix_wrapper.tex