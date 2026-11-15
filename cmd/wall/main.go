package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
)

const usage = `wall predicts the decode tokens/sec ceiling of an LLM on a GPU.

Usage:
  wall [flags]            summary for one GPU, model and batch size
  wall --sweep [flags]    batch sweep 1..256, also writes sweep.csv
  wall compare [flags]    batch-1 ceiling across GPUs and precisions
  wall report [flags]     predicted ceiling vs measured.json

Run "wall <command> -h" for the flags of each command.
`

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintln(os.Stderr, "wall:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	if len(args) > 0 {
		switch args[0] {
		case "compare":
			return runCompare(args[1:], stdout, stderr)
		case "help":
			fmt.Fprint(stdout, usage)
			return nil
		}
	}
	return runCalc(args, stdout, stderr)
}
