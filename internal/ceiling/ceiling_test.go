package ceiling

import (
	"math"
	"testing"
)

func h100Llama8B() Ceilings {
	return Ceilings{
		Memory:  Memory(3350, 16),
		Compute: Compute(989, 16e9),
	}
}

func TestCeilingsH100Llama8B(t *testing.T) {
	c := h100Llama8B()
	if math.Abs(c.Memory-209.375) > 1e-9 {
		t.Errorf("memory ceiling = %v, want ~209", c.Memory)
	}
	if math.Abs(c.Compute-61812.5) > 1e-9 {
		t.Errorf("compute ceiling = %v, want ~61,800", c.Compute)
	}
	if r := c.IdleComputeRatio(); math.Abs(r-295.22) > 0.01 {
		t.Errorf("idle compute ratio = %v, want ~295.2", r)
	}
}

func TestCrossoverBatch(t *testing.T) {
	c := h100Llama8B()
	b := c.CrossoverBatch()
	if b != 296 {
		t.Fatalf("crossover = %d, want 296", b)
	}
	if c.At(b-1).Limit != MemoryBound || c.At(b).Limit != ComputeBound {
		t.Fatalf("limit should flip exactly at batch %d", b)
	}
}

func TestAt(t *testing.T) {
	c := h100Llama8B()
	cases := []struct {
		batch int
		tps   float64
		limit Limit
	}{
		{1, 209.375, MemoryBound},
		{8, 1675, MemoryBound},
		{256, 53600, MemoryBound},
		{512, 61812.5, ComputeBound},
	}
	for _, tc := range cases {
		p := c.At(tc.batch)
		if math.Abs(p.TokensPerSec-tc.tps) > 1e-6 || p.Limit != tc.limit {
			t.Errorf("batch %d: got %.3f %s, want %.3f %s", tc.batch, p.TokensPerSec, p.Limit, tc.tps, tc.limit)
		}
	}
}

func TestSweepBatches(t *testing.T) {
	got := SweepBatches(256)
	if len(got) != 9 || got[0] != 1 || got[8] != 256 {
		t.Fatalf("unexpected sweep batches: %v", got)
	}
}
