# README 

Collection of common packages and definitions for papers and proposals.

## How to set up

Clone this repository inside the paper/

## How to use 

Use the command \input{preamble/<filename>} to load the corresponding packages/definitions.

### Files and their use

common.tex: Includes fixes, pagination, graphics, math, operators, graphTheory, utilities, notations. You will probably always include this.

proposal.tex: Sets the font to times, includes a command to superimpose a grid on a page (useful for checking lengths) and space-saving title and list formatting.

graphicExport.tex: Tools to create PDF files with figures from TikZ or Sketch (uses TikZ export facilities).

graphLocalization.tex: Notation for papers on graph-based localization and control.

generate* : Python scripts that generate the corresponding notation symbols.