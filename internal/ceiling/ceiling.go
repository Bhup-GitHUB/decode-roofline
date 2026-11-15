package ceiling

import (
	"math"

	"github.com/Bhup-GitHUB/decode-roofline/internal/gpu"
	"github.com/Bhup-GitHUB/decode-roofline/internal/model"
)

type Limit string

const (
	MemoryBound  Limit = "memory"
	ComputeBound Limit = "compute"
)

func Memory(bandwidthGBs, weightsGB float64) float64 {
	if weightsGB <= 0 {
		return math.Inf(1)
	}
	return bandwidthGBs / weightsGB
}

func Compute(tflops, flopsPerToken float64) float64 {
	if flopsPerToken <= 0 {
		return math.Inf(1)
	}
	return tflops * 1e12 / flopsPerToken
}

type Ceilings struct {
	Memory  float64
	Compute float64
}

type Point struct {
	Batch        int
	TokensPerSec float64
	Limit        Limit
}

func (c Ceilings) At(batch int) Point {
	if batch < 1 {
		batch = 1
	}
	mem := c.Memory * float64(batch)
	if mem < c.Compute {
		return Point{Batch: batch, TokensPerSec: mem, Limit: MemoryBound}
	}
	return Point{Batch: batch, TokensPerSec: c.Compute, Limit: ComputeBound}
}

func (c Ceilings) IdleComputeRatio() float64 {
	return c.Compute / c.Memory
}

func (c Ceilings) CrossoverBatch() int {
	b := int(math.Ceil(c.IdleComputeRatio()))
	if b < 1 {
		return 1
	}
	return b
}

func SweepBatches(max int) []int {
	var out []int
	for b := 1; b <= max; b *= 2 {
		out = append(out, b)
	}
	return out
}

func (c Ceilings) Sweep(batches []int) []Point {
	points := make([]Point, len(batches))
	for i, b := range batches {
		points[i] = c.At(b)
	}
	return points
}

func For(g gpu.Spec, m model.Model) Ceilings {
	return Ceilings{
		Memory:  Memory(g.BandwidthGBs, m.WeightsGB()),
		Compute: Compute(g.TFLOPS, m.FlopsPerToken()),
	}
}
