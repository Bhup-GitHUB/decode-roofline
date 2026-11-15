package gpu

import (
	"fmt"
	"sort"
	"strings"
)

type Spec struct {
	Name         string
	VRAMGB       float64
	BandwidthGBs float64
	TFLOPS       float64
}

func Table() []Spec {
	return []Spec{
		{Name: "h100-sxm", VRAMGB: 80, BandwidthGBs: 3350, TFLOPS: 989},
		{Name: "h100-pcie", VRAMGB: 80, BandwidthGBs: 2000, TFLOPS: 989},
		{Name: "h200-sxm", VRAMGB: 141, BandwidthGBs: 4800, TFLOPS: 989},
		{Name: "a100-80", VRAMGB: 80, BandwidthGBs: 1935, TFLOPS: 312},
	}
}

func Names() []string {
	table := Table()
	names := make([]string, len(table))
	for i, s := range table {
		names[i] = s.Name
	}
	sort.Strings(names)
	return names
}

func Lookup(name string) (Spec, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	for _, s := range Table() {
		if s.Name == key {
			return s, nil
		}
	}
	return Spec{}, fmt.Errorf("unknown gpu %q (known: %s, or custom)", name, strings.Join(Names(), ", "))
}

func Custom(vramGB, bandwidthGBs, tflops float64) (Spec, error) {
	if vramGB <= 0 || bandwidthGBs <= 0 || tflops <= 0 {
		return Spec{}, fmt.Errorf("custom gpu needs positive --vram, --bandwidth and --tflops")
	}
	return Spec{Name: "custom", VRAMGB: vramGB, BandwidthGBs: bandwidthGBs, TFLOPS: tflops}, nil
}

func (s Spec) String() string {
	return fmt.Sprintf("%s (%g GB, %g GB/s, %g TFLOPS)", s.Name, s.VRAMGB, s.BandwidthGBs, s.TFLOPS)
}
