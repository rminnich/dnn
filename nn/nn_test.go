package nn_test

import (
	"nn/nn"
	"testing"
)

func TestLSB(t *testing.T) {
	wt := []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
	n, err := nn.New(1, .5, wt)
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
func Test2SB(t *testing.T) {
	wt := []float64{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}
	n, err := nn.New(1, .5, wt)
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
		{in: [10]float64{0, .99}, out: false},
		{in: [10]float64{0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, .99}, out: false},
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
func Test4SB(t *testing.T) {
	wt := []float64{0, 0, 0, 0, 1, 1, 1, 1, 0, 0}
	n, err := nn.New(1, .5, wt)
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
		{in: [10]float64{0, .99}, out: false},
		{in: [10]float64{0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, .99}, out: true},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, .99}, out: false},
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
func Test8SB(t *testing.T) {
	wt := []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 1}
	n, err := nn.New(1, .5, wt)
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
		{in: [10]float64{0, .99}, out: false},
		{in: [10]float64{0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, .99}, out: false},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, .99}, out: true},
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

func floatx(f []float64) int {
	var ret int
	for i := range f {
		bit := f[i] > .5
		if bit {
			ret |= (1 << i)
		}
	}
	return ret
}

func TestX(t *testing.T) {
	cols := []nn.ColSpec{
		{Bias: .5, Weight: []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
		{Bias: .5, Weight: []float64{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 1, 1, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
	}
	c, err := nn.NewCol(cols...)
	if err != nil {
		t.Fatalf("NewCol: got %v, want nil", err)
	}
	c.Run()

	for i, tt := range []struct {
		in [10]float64
	}{
		{in: [10]float64{.99}},
		{in: [10]float64{0, .99}},
		{in: [10]float64{0, 0, .99}},
		{in: [10]float64{0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, .99}},
	} {
		// We use go here to simulate lots of async activity.
		// the actual neuron goes in order, and hence will not
		// finish until it has all inputs.
		for i, f := range tt.in {
			go c.Send(uint(i), f)
		}
		of := c.Recv()
		x := floatx(of)
		t.Logf("%d: %v: got %v, %v, want %v", i, tt, of, x, i)

		if x != i {
			t.Errorf("%d: got %v, want %v", i, x, i)
		}

	}
}

func TestArr1x(t *testing.T) {
	cols := []nn.ColSpec{
		{Bias: .5, Weight: []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
		{Bias: .5, Weight: []float64{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 1, 1, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
	}
	n, err := nn.NewNet(cols)
	if err != nil {
		t.Fatalf("NewCol: got %v, want nil", err)
	}
	n.Run()

	for i, tt := range []struct {
		in [10]float64
	}{
		{in: [10]float64{.99}},
		{in: [10]float64{0, .99}},
		{in: [10]float64{0, 0, .99}},
		{in: [10]float64{0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, .99}},
	} {
		// We use go here to simulate lots of async activity.
		// the actual neuron goes in order, and hence will not
		// finish until it has all inputs.
		for i, f := range tt.in {
			go n.Send(uint(i), f)
		}
		of := n.Recv()
		x := floatx(of)
		t.Logf("%d: %v: got %v, %v, want %v", i, tt, of, x, i)

		if x != i {
			t.Errorf("%d: got %v, want %v", i, x, i)
		}

	}
}

func TestArrDecodex(t *testing.T) {
	cols := []nn.ColSpec{
		{Bias: .5, Weight: []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
		{Bias: .5, Weight: []float64{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 1, 1, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
	}
	n, err := nn.NewNet(cols, cols)
	if err != nil {
		t.Fatalf("NewCol: got %v, want nil", err)
	}
	nn.V = t.Logf
	n.Run()

	for i, tt := range []struct {
		in [10]float64
	}{
		{in: [10]float64{.99}},
		{in: [10]float64{0, .99}},
		{in: [10]float64{0, 0, .99}},
		{in: [10]float64{0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, .99}},
		{in: [10]float64{0, 0, 0, 0, 0, 0, 0, 0, 0, .99}},
	} {
		// We use go here to simulate lots of async activity.
		// the actual neuron goes in order, and hence will not
		// finish until it has all inputs.
		for i, f := range tt.in {
			go n.Send(uint(i), f)
		}
		of := n.Recv()
		x := floatx(of)
		t.Logf("%d: %v: got %v, %v, want %v", i, tt, of, x, i)

		if x != i {
			t.Errorf("%d: got %v, want %v", i, x, i)
		}

	}
}

func TestArrTwoColheight1(t *testing.T) {
	cols := []nn.ColSpec{
		{Bias: .5, Weight: []float64{0}},
		//{Bias: .5, Weight: []float64{0, }},
		//{Bias: .5, Weight: []float64{0, }},
		//{Bias: .5, Weight: []float64{0, }},
	}
	n, err := nn.NewNet(cols, cols)
	if err != nil {
		t.Fatalf("NewCol: got %v, want nil", err)
	}
	nn.V = t.Logf
	n.Run()

	for i, tt := range []struct {
		in [1]float64
	}{
		{in: [1]float64{.99}},
	} {
		// We use go here to simulate lots of async activity.
		// the actual neuron goes in order, and hence will not
		// finish until it has all inputs.
		t.Logf("Send %d things", len(tt.in))
		for i, f := range tt.in {
			go n.Send(uint(i), f)
		}
		t.Logf("Recv...")
		of := n.Recv()
		x := floatx(of)
		t.Logf("%d: %v: got %v, %v, want %v", i, tt, of, x, i)

		if x != i {
			t.Errorf("%d: got %v, want %v", i, x, i)
		}

	}
}

func TestArrTwoColHeight2(t *testing.T) {
	cols := []nn.ColSpec{
		{Bias: .5, Weight: []float64{0, 0}},
		{Bias: .5, Weight: []float64{0, 0}},
		//{Bias: .5, Weight: []float64{0, }},
		//{Bias: .5, Weight: []float64{0, }},
	}
	n, err := nn.NewNet(cols, cols)
	if err != nil {
		t.Fatalf("NewCol: got %v, want nil", err)
	}
	nn.V = t.Logf
	n.Run()

	for i, tt := range []struct {
		in [2]float64
	}{
		{in: [2]float64{.99}},
	} {
		// We use go here to simulate lots of async activity.
		// the actual neuron goes in order, and hence will not
		// finish until it has all inputs.
		t.Logf("Send %d things", len(tt.in))
		for i, f := range tt.in {
			go n.Send(uint(i), f)
		}
		t.Logf("Recv...")
		of := n.Recv()
		x := floatx(of)
		t.Logf("%d: %v: got %v, %v, want %v", i, tt, of, x, i)

		if x != i {
			t.Errorf("%d: got %v, want %v", i, x, i)
		}

	}
}

func TestArrTwoColHeight4(t *testing.T) {
	cols := []nn.ColSpec{
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
	}
	n, err := nn.NewNet(cols, cols)
	if err != nil {
		t.Fatalf("NewCol: got %v, want nil", err)
	}
	nn.V = t.Logf
	n.Run()

	for i, tt := range []struct {
		in [4]float64
	}{
		{in: [4]float64{.99}},
	} {
		// We use go here to simulate lots of async activity.
		// the actual neuron goes in order, and hence will not
		// finish until it has all inputs.
		t.Logf("Send %d things", len(tt.in))
		for i, f := range tt.in {
			go n.Send(uint(i), f)
		}
		t.Logf("Recv...")
		of := n.Recv()
		x := floatx(of)
		t.Logf("%d: %v: got %v, %v, want %v", i, tt, of, x, i)

		if x != i {
			t.Errorf("%d: got %v, want %v", i, x, i)
		}

	}
}
