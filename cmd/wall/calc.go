package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/ceiling"
	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/gpu"
	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/model"
	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/report"
)

const maxSweepBatch = 256

func runCalc(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("wall", flag.ContinueOnError)
	fs.SetOutput(stderr)

	var hw hardwareFlags
	hw.register(fs)
	batch := fs.Int("batch", 1, "number of concurrent sequences")
	context := fs.Int("context", 0, "tokens of context per sequence, for the KV cache estimate")
	layers := fs.Int("layers", 0, "transformer layers, for the KV cache estimate")
	hidden := fs.Int("hidden", 0, "hidden size, for the KV cache estimate")
	sweep := fs.Bool("sweep", false, "print batch sizes 1..256 and write a CSV")
	csvPath := fs.String("csv", "sweep.csv", "where --sweep writes its CSV")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *batch < 1 {
		return fmt.Errorf("--batch must be at least 1")
	}

	g, m, err := hw.resolve()
	if err != nil {
		return err
	}
	c := ceiling.For(g, m)

	if m.WeightsGB() >= g.VRAMGB {
		fmt.Fprintf(stderr, "warning: %.1f GB of weights does not fit in %g GB of VRAM on %s\n\n", m.WeightsGB(), g.VRAMGB, g.Name)
	}

	if *sweep {
		return writeSweep(stdout, g, m, *csvPath)
	}

	s := report.Summary{
		GPU:      g,
		Model:    m,
		KV:       model.KVShape{Layers: *layers, Hidden: *hidden, Context: *context, Batch: *batch},
		Ceilings: c,
		Batch:    *batch,
	}
	return s.Print(stdout)
}

func writeSweep(w io.Writer, g gpu.Spec, m model.Model, path string) error {
	c := ceiling.For(g, m)
	fmt.Fprintf(w, "%.3gB params at %s on %s\n\n", m.Params/1e9, m.Precision, g.Name)
	points := c.Sweep(ceiling.SweepBatches(maxSweepBatch))
	if err := report.PrintSweep(w, points); err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := report.WriteSweepCSV(f, points); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	fmt.Fprintf(w, "\ncrossover at batch %d, wrote %s\n", c.CrossoverBatch(), path)
	report.Footer(w)
	return nil
}
