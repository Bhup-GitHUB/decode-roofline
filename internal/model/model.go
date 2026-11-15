package model

import (
	"fmt"
	"strings"
)

type Precision string

const (
	FP16 Precision = "fp16"
	FP8  Precision = "fp8"
	INT8 Precision = "int8"
	INT4 Precision = "int4"
)

const kvBytesPerValue = 2

func Precisions() []Precision {
	return []Precision{FP16, FP8, INT8, INT4}
}

func ParsePrecision(s string) (Precision, error) {
	p := Precision(strings.ToLower(strings.TrimSpace(s)))
	for _, known := range Precisions() {
		if p == known {
			return p, nil
		}
	}
	return "", fmt.Errorf("unknown precision %q (want fp16, fp8, int8 or int4)", s)
}

func (p Precision) BytesPerWeight() float64 {
	switch p {
	case FP16:
		return 2
	case FP8, INT8:
		return 1
	case INT4:
		return 0.5
	}
	return 0
}

type Model struct {
	Params    float64
	Precision Precision
}

func (m Model) WeightsGB() float64 {
	return m.Params * m.Precision.BytesPerWeight() / 1e9
}

func (m Model) FlopsPerToken() float64 {
	return 2 * m.Params
}

type KVShape struct {
	Layers  int
	Hidden  int
	Context int
	Batch   int
}

func (k KVShape) Valid() bool {
	return k.Layers > 0 && k.Hidden > 0 && k.Context > 0 && k.Batch > 0
}

func (k KVShape) GB() float64 {
	if !k.Valid() {
		return 0
	}
	perToken := 2 * float64(k.Layers) * float64(k.Hidden) * kvBytesPerValue
	return perToken * float64(k.Context) * float64(k.Batch) / 1e9
}
