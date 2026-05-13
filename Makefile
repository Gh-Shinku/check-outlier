.PHONY: run test resume

run:
	@python main.py

test:
	@python main.py -n 1

resume:
	@python main.py --resume
