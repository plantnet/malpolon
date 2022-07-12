SOURCEDIR = malpolon

type_check:
	dmypy run

style_check:
	black --check $(SOURCEDIR)

tests:
	pytest malpolon/tests

coverage:
	pytest --cov malpolon/
