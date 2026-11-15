GPU       ?= h100-sxm
PARAMS    ?= 8e9
PRECISION ?= fp16
BATCH     ?= 1

ARGS = --gpu $(GPU) --params $(PARAMS) --precision $(PRECISION)

.PHONY: run sweep compare report test build bench

run:
	go run ./cmd/wall $(ARGS) --batch $(BATCH)

sweep:
	go run ./cmd/wall $(ARGS) --sweep

compare:
	go run ./cmd/wall compare $(ARGS)

report:
	go run ./cmd/wall report $(ARGS)

test:
	go vet ./...
	go test ./...

build:
	go build -o bin/wall ./cmd/wall
