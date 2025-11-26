.PHONY: all clean clean-all

# Variables
PRESENTATION = presentation
TEX_FILE = $(PRESENTATION).tex
PDF_FILE = $(PRESENTATION).pdf
BIB_FILE = refs.bib

# Intermediate files to be cleaned
AUX_FILES = *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk *.synctex.gz *.dvi *.ps *.bbl *.bcf *.blg *.run.xml

# Number of CPU cores for parallel processing
CORES := $(shell nproc)

# Check if biber is available
BIBER := $(shell which biber 2>/dev/null)

all: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE) $(BIB_FILE)
	@echo "Building PDF with $(CORES) cores..."
	pdflatex -interaction=nonstopmode -shell-escape $(TEX_FILE)
ifdef BIBER
	@echo "Running biber for bibliography..."
	biber $(PRESENTATION)
	pdflatex -interaction=nonstopmode -shell-escape $(TEX_FILE)
	pdflatex -interaction=nonstopmode -shell-escape $(TEX_FILE)
else
	@echo "Warning: biber not found, citations will not be resolved"
	@echo "Install biber with: sudo pacman -S biber (or your package manager)"
	pdflatex -interaction=nonstopmode -shell-escape $(TEX_FILE)
endif
	@echo "Build complete! PDF: $(PDF_FILE)"

clean:
	rm -f $(AUX_FILES)

clean-all: clean
	rm -f $(PDF_FILE)

.DEFAULT_GOAL := all
