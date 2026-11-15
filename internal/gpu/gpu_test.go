package gpu

import "testing"

func TestLookup(t *testing.T) {
	s, err := Lookup("H100-SXM")
	if err != nil {
		t.Fatal(err)
	}
	if s.BandwidthGBs != 3350 || s.TFLOPS != 989 || s.VRAMGB != 80 {
		t.Fatalf("unexpected spec: %+v", s)
	}
	if _, err := Lookup("rtx-9090"); err == nil {
		t.Fatal("expected error for unknown gpu")
	}
}

func TestCustomRejectsMissingValues(t *testing.T) {
	if _, err := Custom(80, 0, 989); err == nil {
		t.Fatal("expected error when bandwidth is zero")
	}
}
