package report

import (
	"fmt"
	"io"

	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/ceiling"
	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/gpu"
	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/model"
)

func ceilingsFor(g gpu.Spec, m model.Model) ceiling.Ceilings {
	return ceiling.Ceilings{
		Memory:  ceiling.Memory(g.BandwidthGBs, m.WeightsGB()),
		Compute: ceiling.Compute(g.TFLOPS, m.FlopsPerToken()),
	}
}

func fitLabel(g gpu.Spec, m model.Model) string {
	if m.WeightsGB() < g.VRAMGB {
		return "yes"
	}
	return "no"
}

func CompareGPUs(w io.Writer, gpus []gpu.Spec, m model.Model) error {
	fmt.Fprintf(w, "%s params at %s, batch 1, across GPUs\n\n", params(m.Params), m.Precision)
	tw := newTable(w)
	fmt.Fprintln(tw, "GPU\tGB/S\tWEIGHTS\tFITS\tMEMORY CEIL\tCOMPUTE CEIL\tBATCH-1 TOK/S")
	for _, g := range gpus {
		c := ceilingsFor(g, m)
		fmt.Fprintf(tw, "%s\t%g\t%s\t%s\t%s\t%s\t%s\n",
			g.Name, g.BandwidthGBs, gb(m.WeightsGB()), fitLabel(g, m),
			tps(c.Memory), tps(c.Compute), tps(c.At(1).TokensPerSec))
	}
	return tw.Flush()
}

func ComparePrecisions(w io.Writer, g gpu.Spec, paramCount float64) error {
	fmt.Fprintf(w, "%s params on %s, batch 1, across precisions\n\n", params(paramCount), g.Name)
	tw := newTable(w)
	fmt.Fprintln(tw, "PRECISION\tBYTES/W\tWEIGHTS\tFITS\tMEMORY CEIL\tCROSSOVER\tBATCH-1 TOK/S")
	for _, p := range model.Precisions() {
		m := model.Model{Params: paramCount, Precision: p}
		c := ceilingsFor(g, m)
		fmt.Fprintf(tw, "%s\t%g\t%s\t%s\t%s\t%d\t%s\n",
			p, p.BytesPerWeight(), gb(m.WeightsGB()), fitLabel(g, m),
			tps(c.Memory), c.CrossoverBatch(), tps(c.At(1).TokensPerSec))
	}
	return tw.Flush()
}
