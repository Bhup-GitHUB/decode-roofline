package model

import (
	"math"
	"testing"
)

func near(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestWeightsGB(t *testing.T) {
	cases := []struct {
		p    Precision
		want float64
	}{
		{FP16, 16},
		{FP8, 8},
		{INT8, 8},
		{INT4, 4},
	}
	for _, c := range cases {
		m := Model{Params: 8e9, Precision: c.p}
		if got := m.WeightsGB(); !near(got, c.want, 1e-9) {
			t.Errorf("%s: weights = %v GB, want %v", c.p, got, c.want)
		}
	}
}

func TestFlopsPerToken(t *testing.T) {
	m := Model{Params: 8e9, Precision: FP16}
	if got := m.FlopsPerToken(); got != 16e9 {
		t.Fatalf("flops per token = %v, want 16e9", got)
	}
}

func TestKVCacheGB(t *testing.T) {
	k := KVShape{Layers: 32, Hidden: 4096, Context: 8192, Batch: 1}
	if got := k.GB(); !near(got, 4.294967296, 1e-9) {
		t.Fatalf("kv cache = %v GB, want ~4.29", got)
	}
	k.Batch = 0
	if got := k.GB(); got != 0 {
		t.Fatalf("incomplete shape should give 0, got %v", got)
	}
}

func TestParsePrecision(t *testing.T) {
	if p, err := ParsePrecision("FP8"); err != nil || p != FP8 {
		t.Fatalf("got %q, %v", p, err)
	}
	if _, err := ParsePrecision("bf16"); err == nil {
		t.Fatal("expected error for bf16")
	}
}
