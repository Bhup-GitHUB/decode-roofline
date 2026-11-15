package main

import (
	"flag"
	"fmt"
	"io"

	"github.com/Bhup-GitHUB/decode-roofline/internal/gpu"
	"github.com/Bhup-GitHUB/decode-roofline/internal/report"
)

func runCompare(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("wall compare", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var hw hardwareFlags
	hw.register(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}

	g, m, err := hw.resolve()
	if err != nil {
		return err
	}

	if err := report.CompareGPUs(stdout, gpu.Table(), m); err != nil {
		return err
	}
	fmt.Fprintln(stdout)
	if err := report.ComparePrecisions(stdout, g, m.Params); err != nil {
		return err
	}
	report.Footer(stdout)
	return nil
}
