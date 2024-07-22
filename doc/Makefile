DOC := project.tex

all: doc

doc: $(DOC:.tex=.pdf)

all: project.pdf

%.pdf: %.tex
	pdflatex -shell-escape $< && bibtex $* && \
	pdflatex -shell-escape $< && pdflatex -shell-escape $<

clean:
	rm  project.pdf *.aux *.blg *.log *.bbl *.toc *.out *.glg *.glo \
					*.gls *.ist *.lof *.lot *.fls *synctex.gz *.fdb_latexmk