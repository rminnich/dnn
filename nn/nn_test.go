package nn_test

import (
	"nn/nn"
	"testing"
)

func TestLSB(t *testing.T) {
	wt := []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
	n, err := nn.New(wt, .5, 1)
	n.Run()
	if err != nil {
		t.Fatalf("New: got %v, want nil", err)
	}

	for i, tt := range []struct {
		in  [10]float64
		out bool
	}{
		{in: [10]float64{0}, out: false},
		{in: [10]float64{.99}, out: false},
		{in: [10]float64{0, .99}, out: true},
		{in: [10]float64{0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, .99}, out: true},
	} {
		// We use go here to simulate lots of async activity.
		// the actual neuron goes in order, and hence will not
		// finish until it has all inputs.
		for i, f := range tt.in {
			go n.Send(uint(i), f)
		}
		of := n.Recv(0)
		t.Logf("%d: %v: result %v > .5 %v want %v", i, tt, of, of > .5, tt.out)
		if of > .5 != tt.out {
			t.Errorf("%d: got %v, test %v, want %v", i, of, of > .5, tt.out)
		}

	}
}
