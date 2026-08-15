# VITA-E ECCV-to-IEEE Transfer Report

## Target Assumption

The IEEE manuscript is prepared as a 2026 regular journal article for *IEEE Transactions on Circuits and Systems for Video Technology* (TCSVT). If the intended journal or article type differs, the running header and venue-specific requirements must be updated.

## Content Preserved

- The paper title, all 18 authors in their original order, and the four original institution names.
- The abstract and all sections used by `eccv2026_conference.tex`: Introduction, Related Work, VITA-E System Architecture, Experiments and Results, and Conclusion and Future Work.
- The acknowledgments text present in the ECCV source tree.
- The complete BibTeX database and all six graphics referenced by the active manuscript.
- All main-paper prose, equations, table contents, numerical results, captions, labels, and citation commands, except for the user-requested removal of the supplementary-material sentence from the Fig. 1 caption. Section-file comparison otherwise shows only IEEE layout changes to float environments, widths, and spacing.

## Missing IEEE Metadata and Placeholders

| Item | Current handling |
|---|---|
| Index terms / keywords | Visible placeholder in `IEEEkeywords`; the ECCV source supplies none. |
| Complete postal addresses | Visible placeholder in the first-page author footnotes. |
| Author e-mail addresses | Visible placeholder in the first-page author footnotes. |
| Corresponding author | Visible placeholder in the first-page author footnotes. |
| IEEE membership grades | `TODO` comment in the author block; no membership data appears in the ECCV source. |
| Funding / grant information | `TODO` comment in the author block; no funding data appears in the ECCV source. |
| ORCID identifiers | Not inserted; none appear in the ECCV source. These may also be requested by the submission system. |
| Author biographies and photos | Omitted because the supplied IEEE template marks them as optional and the ECCV source contains neither. |
| Manuscript received/revised dates and IEEE publication ID | Omitted for submission; these are production metadata added after acceptance. |

## ECCV Materials Not Migrated

- `VITA-E/sec/conclusion.tex`: not referenced by the active ECCV main file; the active paper uses `discussion.tex` instead.
- The separately compiled ECCV appendix and `VITA-E/sec/appendix.tex`: excluded from the IEEE manuscript at the user's request.
- The Fig. 1 caption sentence directing readers to a supplementary demo video: removed at the user's request.
- ECCV/LLNCS class and style files, ECCV review ID and running-title metadata, and the legacy ICLR entry file/style.
- ECCV build artifacts and generated PDFs (`.aux`, `.bbl`, `.blg`, `.log`, `.fls`, `.fdb_latexmk`, and similar files).
- `VITA-E/demo/`: runtime/demo code rather than manuscript content.
- `VITA-E/pic/libero0.pdf`: not referenced by the active ECCV main manuscript.

Commented homepage and YouTube-link lines remain in the copied section sources but remain invisible in the compiled paper, matching the ECCV source.

## IEEE Layout Adaptations

- Replaced the ECCV/LLNCS document wrapper with `IEEEtran` journal mode and IEEE numeric bibliography style.
- Mapped authors to their four original affiliations using IEEE first-page footnotes.
- Set Figs. 2 and 4 as full-width double-column floats using `figure*` and `\textwidth`; kept Fig. 3 as a centered double-column float at `0.8\textwidth`.
- Kept Figs. 1, 5, and 6 as single-column floats and expanded each to the full column width using `\linewidth`.
- Kept Tables 1, 2, 3, and 4 as single-column floats and expanded each to the full column width using `\linewidth`.
- Removed ECCV negative spacing around Tables 1--4 to prevent caption/table overlap in the IEEE layout.
- Removed the duplicate heading command from the acknowledgments fragment because the IEEE main file supplies the required unnumbered heading.

## Validation

- Compiled successfully with MacTeX 2026 using `latexmk`, `pdflatex`, and BibTeX.
- Output: 9 pages, US Letter, IEEE two-column journal layout.
- No overfull boxes, missing graphics, undefined cross-references, or unresolved citations in the final build.
- All 9 pages were rendered to PNG and visually checked for clipping, overlap, table overflow, float order, headers, and page numbering.
- The copied math-command file, BibTeX database, and all six graphics are byte-identical to the ECCV sources.

## Submission Notes Not Changed in the Manuscript

- The current TCSVT manuscript guide requires that the work not be concurrently submitted elsewhere. If this is an expanded version of a published conference paper, the guide also requires citation of the earlier work, non-identical content, and novel elements; this transfer did not modify the paper to address those points.
- The source bibliography contains five unused entries (`Bengio+chapter2007`, `Hinton06`, `goodfellow2016deep`, `Radford2022RobustSR`, and `Du2024CosyVoiceAS`). The last two were cited only by the excluded appendix. All were retained to keep the BibTeX database unchanged and do not appear in the PDF.

Official pages checked on 2026-08-05:

- <https://ieee-cas.org/publication/tcsvt/guidelines-authors>
- <https://ieee-cas.org/publication/tcsvt/tcsvt-manuscript-submission-guide>
- <https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/>
