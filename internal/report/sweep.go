package report

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"

	"github.com/Bhup-GitHUB/decode-roofline/internal/ceiling"
)

func PrintSweep(w io.Writer, points []ceiling.Point) error {
	tw := newTable(w)
	fmt.Fprintln(tw, "BATCH\tTOK/S\tPER SEQ\tLIMIT")
	for _, p := range points {
		fmt.Fprintf(tw, "%d\t%s\t%s\t%s\n", p.Batch, tps(p.TokensPerSec), tps(p.TokensPerSec/float64(p.Batch)), p.Limit)
	}
	return tw.Flush()
}

func WriteSweepCSV(w io.Writer, points []ceiling.Point) error {
	cw := csv.NewWriter(w)
	if err := cw.Write([]string{"batch", "tokens_per_sec", "tokens_per_sec_per_seq", "limit"}); err != nil {
		return err
	}
	for _, p := range points {
		row := []string{
			strconv.Itoa(p.Batch),
			strconv.FormatFloat(p.TokensPerSec, 'f', 2, 64),
			strconv.FormatFloat(p.TokensPerSec/float64(p.Batch), 'f', 2, 64),
			string(p.Limit),
		}
		if err := cw.Write(row); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
