package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/gpu"
	"github.com/Bhup-GitHUB/gpu-memory-wall/internal/model"
)

type hardwareFlags struct {
	gpu       string
	vram      float64
	bandwidth float64
	tflops    float64
	params    float64
	precision string
}

func (h *hardwareFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&h.gpu, "gpu", "h100-sxm", "gpu name ("+strings.Join(gpu.Names(), ", ")+", or custom)")
	fs.Float64Var(&h.vram, "vram", 0, "custom gpu VRAM in GB")
	fs.Float64Var(&h.bandwidth, "bandwidth", 0, "custom gpu memory bandwidth in GB/s")
	fs.Float64Var(&h.tflops, "tflops", 0, "custom gpu dense FP16 TFLOPS")
	fs.Float64Var(&h.params, "params", 0, "model parameter count, e.g. 8e9")
	fs.StringVar(&h.precision, "precision", "fp16", "weight precision: fp16, fp8, int8, int4")
}

func (h *hardwareFlags) spec() (gpu.Spec, error) {
	if strings.EqualFold(h.gpu, "custom") {
		return gpu.Custom(h.vram, h.bandwidth, h.tflops)
	}
	return gpu.Lookup(h.gpu)
}

func (h *hardwareFlags) model() (model.Model, error) {
	if h.params <= 0 {
		return model.Model{}, fmt.Errorf("--params is required, e.g. --params 8e9")
	}
	p, err := model.ParsePrecision(h.precision)
	if err != nil {
		return model.Model{}, err
	}
	return model.Model{Params: h.params, Precision: p}, nil
}

func (h *hardwareFlags) resolve() (gpu.Spec, model.Model, error) {
	g, err := h.spec()
	if err != nil {
		return gpu.Spec{}, model.Model{}, err
	}
	m, err := h.model()
	if err != nil {
		return gpu.Spec{}, model.Model{}, err
	}
	return g, m, nil
}
