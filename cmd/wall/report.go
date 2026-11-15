package main

import (
	"flag"
	"fmt"
	"io"

	"github.com/Bhup-GitHUB/decode-roofline/internal/ceiling"
	"github.com/Bhup-GitHUB/decode-roofline/internal/report"
)

func runReport(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("wall report", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var hw hardwareFlags
	hw.register(fs)
	path := fs.String("measured", "measured.json", "file written by bench/bench_vllm.py")
	if err := fs.Parse(args); err != nil {
		return err
	}

	g, m, err := hw.resolve()
	if err != nil {
		return err
	}
	measured, err := report.LoadMeasured(*path)
	if err != nil {
		return err
	}

	fmt.Fprintf(stdout, "%s on %s, %s weights, max_tokens %d, median of %d runs\n\n",
		measured.Model, g.Name, m.Precision, measured.MaxTokens, measured.Repeats)
	rows := report.Compare(ceiling.For(g, m), measured)
	if err := report.PrintComparison(stdout, rows); err != nil {
		return err
	}
	report.Footer(stdout)
	return nil
}
