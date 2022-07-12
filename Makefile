SOURCEDIR = malpolon

type_check:
	dmypy run

style_check:
	black --check $(SOURCEDIR)
