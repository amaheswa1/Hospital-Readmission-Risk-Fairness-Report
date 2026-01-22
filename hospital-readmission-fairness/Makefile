.PHONY: validate train report test run

validate:
	readmission validate

train:
	readmission train

report:
	readmission report

test:
	pytest -q

run: validate train report
