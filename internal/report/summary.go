package report

import (
	"fmt"
	"io"

	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/ceiling"
	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/gpu"
	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/model"
)

type Summary struct {
	GPU      gpu.Spec
	Model    model.Model
	KV       model.KVShape
	Ceilings ceiling.Ceilings
	Batch    int
}

func (s Summary) Fits() bool {
	return s.Model.WeightsGB() < s.GPU.VRAMGB
}

func (s Summary) Print(w io.Writer) error {
	weights := s.Model.WeightsGB()
	fit := "fits"
	if !s.Fits() {
		fit = "DOES NOT FIT"
	}
	at := s.Ceilings.At(s.Batch)

	tw := newTable(w)
	fmt.Fprintf(tw, "GPU\t%s\n", s.GPU)
	fmt.Fprintf(tw, "Model\t%s params, %s (%g bytes/weight)\n", params(s.Model.Params), s.Model.Precision, s.Model.Precision.BytesPerWeight())
	fmt.Fprintf(tw, "Weights\t%s (%s in %g GB VRAM)\n", gb(weights), fit, s.GPU.VRAMGB)
	if s.KV.Valid() {
		kv := s.KV.GB()
		fmt.Fprintf(tw, "KV cache\t%s (%d seq x %d tokens, fp16)\n", gb(kv), s.KV.Batch, s.KV.Context)
		fmt.Fprintf(tw, "Weights + KV\t%s of %g GB\n", gb(weights+kv), s.GPU.VRAMGB)
	}
	fmt.Fprintf(tw, "\t\n")
	fmt.Fprintf(tw, "Memory ceiling\t%s tok/s per sequence\n", tps(s.Ceilings.Memory))
	fmt.Fprintf(tw, "FLOPs per token\t%.3g\n", s.Model.FlopsPerToken())
	fmt.Fprintf(tw, "Compute ceiling\t%s tok/s total\n", tps(s.Ceilings.Compute))
	fmt.Fprintf(tw, "Idle compute ratio\t%.1fx at batch 1\n", s.Ceilings.IdleComputeRatio())
	fmt.Fprintf(tw, "Crossover batch\t%d\n", s.Ceilings.CrossoverBatch())
	fmt.Fprintf(tw, "\t\n")
	fmt.Fprintf(tw, "Batch %d\t%s tok/s, %s bound\n", at.Batch, tps(at.TokensPerSec), at.Limit)
	if err := tw.Flush(); err != nil {
		return err
	}
	Footer(w)
	return nil
}
