# Compile the assignment report using the same course image on every platform.
CST463_IMAGE ?= samogden/cst463:latest
REPORT_COMPILER ?= cst463-compile-report

.PHONY: compile-report compile-report-html check-report

ifeq ($(CST463_IN_CONTAINER),1)
compile-report:
	$(REPORT_COMPILER) report.md report.pdf

compile-report-html:
	$(REPORT_COMPILER) --format html report.md report.html

check-report:
	$(REPORT_COMPILER) --check report.md
else
ifeq ($(OS),Windows_NT)
DOCKER_USER :=
else
DOCKER_USER := --user $(shell id -u):$(shell id -g)
endif

compile-report:
	docker run --rm $(DOCKER_USER) -v "$(CURDIR):/workspace" -w /workspace "$(CST463_IMAGE)" $(REPORT_COMPILER) report.md report.pdf

compile-report-html:
	docker run --rm $(DOCKER_USER) -v "$(CURDIR):/workspace" -w /workspace "$(CST463_IMAGE)" $(REPORT_COMPILER) --format html report.md report.html

check-report:
	docker run --rm $(DOCKER_USER) -v "$(CURDIR):/workspace" -w /workspace "$(CST463_IMAGE)" $(REPORT_COMPILER) --check report.md
endif
