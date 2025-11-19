.PHONY: all clean

# Variables
PRESENTATION = presentation
TEX_FILE = $(PRESENTATION).tex
PDF_FILE = $(PRESENTATION).pdf

# Intermediate files to be cleaned
AUX_FILES = *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz *.dvi *.ps

all: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE)
	pdflatex -interaction=nonstopmode $(TEX_FILE)
	pdflatex -interaction=nonstopmode $(TEX_FILE)

clean:
	rm -f $(AUX_FILES)

clean-all: clean
	rm -f $(PDF_FILE)

.DEFAULT_GOAL := all
