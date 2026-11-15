package report

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/ceiling"
)

func TestGroupThousands(t *testing.T) {
	cases := map[string]string{
		"61813":   "61,813",
		"1000000": "1,000,000",
		"999":     "999",
		"-1234.5": "-1,234.5",
	}
	for in, want := range cases {
		if got := groupThousands(in); got != want {
			t.Errorf("groupThousands(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestWriteSweepCSV(t *testing.T) {
	c := ceiling.Ceilings{Memory: 209.375, Compute: 61812.5}
	var buf bytes.Buffer
	if err := WriteSweepCSV(&buf, c.Sweep([]int{1, 512})); err != nil {
		t.Fatal(err)
	}
	want := "batch,tokens_per_sec,tokens_per_sec_per_seq,limit\n" +
		"1,209.38,209.38,memory\n" +
		"512,61812.50,120.73,compute\n"
	if buf.String() != want {
		t.Fatalf("csv mismatch:\n%s", buf.String())
	}
}

func TestLoadAndCompareMeasured(t *testing.T) {
	path := filepath.Join(t.TempDir(), "measured.json")
	data := `{"model":"llama","max_tokens":256,"repeats":3,
		"single_stream":{"decode_tokens_per_sec":167.5},
		"concurrency":[{"concurrency":8,"output_tokens_per_sec":1340}]}`
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadMeasured(path)
	if err != nil {
		t.Fatal(err)
	}
	rows := Compare(ceiling.Ceilings{Memory: 209.375, Compute: 61812.5}, m)
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want 2", len(rows))
	}
	if p := rows[0].Percent(); p < 79.9 || p > 80.1 {
		t.Errorf("single stream percent = %.2f, want 80", p)
	}
	if p := rows[1].Percent(); p < 79.9 || p > 80.1 {
		t.Errorf("concurrency 8 percent = %.2f, want 80", p)
	}

	var out bytes.Buffer
	if err := PrintComparison(&out, rows); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "80.0%") {
		t.Fatalf("table missing percent:\n%s", out.String())
	}
}
