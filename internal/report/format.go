package report

import (
	"fmt"
	"io"
	"math"
	"strings"
	"text/tabwriter"
)

func newTable(w io.Writer) *tabwriter.Writer {
	return tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
}

func groupThousands(s string) string {
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	intPart, frac, hasFrac := strings.Cut(s, ".")
	var b strings.Builder
	for i, r := range intPart {
		if i > 0 && (len(intPart)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(r)
	}
	out := b.String()
	if hasFrac {
		out += "." + frac
	}
	if neg {
		out = "-" + out
	}
	return out
}

func tps(v float64) string {
	if math.IsInf(v, 0) || math.IsNaN(v) {
		return "n/a"
	}
	if v >= 1000 {
		return groupThousands(fmt.Sprintf("%.0f", v))
	}
	return fmt.Sprintf("%.1f", v)
}

func gb(v float64) string {
	return fmt.Sprintf("%.2f GB", v)
}

func params(p float64) string {
	switch {
	case p >= 1e12:
		return fmt.Sprintf("%.2gT", p/1e12)
	case p >= 1e9:
		return fmt.Sprintf("%.3gB", p/1e9)
	case p >= 1e6:
		return fmt.Sprintf("%.3gM", p/1e6)
	}
	return fmt.Sprintf("%.0f", p)
}

func Footer(w io.Writer) {
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Assumptions:")
	fmt.Fprintln(w, "  - all weights are read from HBM once per decode step")
	fmt.Fprintln(w, "  - KV cache traffic and all software overheads are ignored")
	fmt.Fprintln(w, "  - rated peak memory bandwidth and dense FP16 TFLOPS are reached")
	fmt.Fprintln(w, "These are theoretical upper bounds, not forecasts.")
}
