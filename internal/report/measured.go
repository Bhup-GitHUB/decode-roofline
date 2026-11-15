package report

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/ceiling"
)

type SingleStream struct {
	DecodeTokensPerSec float64 `json:"decode_tokens_per_sec"`
	TTFTMillis         float64 `json:"ttft_ms"`
}

type ConcurrencyRun struct {
	Concurrency        int     `json:"concurrency"`
	OutputTokensPerSec float64 `json:"output_tokens_per_sec"`
}

type Measured struct {
	Model        string           `json:"model"`
	BaseURL      string           `json:"base_url"`
	MaxTokens    int              `json:"max_tokens"`
	Repeats      int              `json:"repeats"`
	SingleStream SingleStream     `json:"single_stream"`
	Concurrency  []ConcurrencyRun `json:"concurrency"`
}

func LoadMeasured(path string) (Measured, error) {
	var m Measured
	f, err := os.Open(path)
	if err != nil {
		return m, err
	}
	defer f.Close()
	if err := json.NewDecoder(f).Decode(&m); err != nil {
		return m, fmt.Errorf("parse %s: %w", path, err)
	}
	if len(m.Concurrency) == 0 && m.SingleStream.DecodeTokensPerSec == 0 {
		return m, fmt.Errorf("%s has no measurements", path)
	}
	return m, nil
}

type Comparison struct {
	Label     string
	Predicted ceiling.Point
	Measured  float64
}

func (c Comparison) Percent() float64 {
	if c.Predicted.TokensPerSec == 0 {
		return 0
	}
	return 100 * c.Measured / c.Predicted.TokensPerSec
}

func Compare(c ceiling.Ceilings, m Measured) []Comparison {
	var rows []Comparison
	if m.SingleStream.DecodeTokensPerSec > 0 {
		rows = append(rows, Comparison{
			Label:     "1 (decode only)",
			Predicted: c.At(1),
			Measured:  m.SingleStream.DecodeTokensPerSec,
		})
	}
	for _, r := range m.Concurrency {
		rows = append(rows, Comparison{
			Label:     fmt.Sprint(r.Concurrency),
			Predicted: c.At(r.Concurrency),
			Measured:  r.OutputTokensPerSec,
		})
	}
	return rows
}

func PrintComparison(w io.Writer, m Measured, rows []Comparison) error {
	fmt.Fprintf(w, "model %s, max_tokens %d, median of %d runs\n\n", m.Model, m.MaxTokens, m.Repeats)
	tw := newTable(w)
	fmt.Fprintln(tw, "CONCURRENCY\tPREDICTED CEIL\tLIMIT\tMEASURED\t% OF CEIL")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%.1f%%\n",
			r.Label, tps(r.Predicted.TokensPerSec), r.Predicted.Limit, tps(r.Measured), r.Percent())
	}
	return tw.Flush()
}
