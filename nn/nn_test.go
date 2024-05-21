package nn_test

import (
	"errors"
	"nn/nn"
	"os"
	"testing"
)

func TestLSB(t *testing.T) {
	wt := []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}
	n, err := nn.New("0", 1, .5, wt)
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
	n, err := nn.New("0", 1, .5, wt)
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
	n, err := nn.New("0", 1, .5, wt)
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
	n, err := nn.New("0", 1, .5, wt)
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
	layers := []nn.LayerSpec{
		{Bias: .5, Weight: []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
		{Bias: .5, Weight: []float64{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 1, 1, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
	}
	c, err := nn.NewLayer("0", layers...)
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
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
	layers := []nn.LayerSpec{
		{Bias: .5, Weight: []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
		{Bias: .5, Weight: []float64{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 1, 1, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
	}
	c, err := nn.NewLayer("0", layers...)
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
	}
	t.Logf("Direct layers %v", c)
	n, err := nn.NewNet(layers)
	nn.V = t.Logf
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
	}
	t.Logf("Layers via net: %v", n)
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
		t.Logf("======================== %d ==================", i)
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

func TestArrBadTopology(t *testing.T) {
	layers := []nn.LayerSpec{
		{Bias: .5, Weight: []float64{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}},
		{Bias: .5, Weight: []float64{0, 0, 1, 1, 0, 0, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 1, 1, 1, 1, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 1}},
	}
	if _, err := nn.NewNet(layers, layers); !errors.Is(err, os.ErrInvalid) {
		t.Fatalf("NewLayer: got nil, want %v", os.ErrInvalid)
	}
}

func TestArrTwoLayerheight1(t *testing.T) {
	layers := []nn.LayerSpec{
		{Bias: .5, Weight: []float64{0}},
		//{Bias: .5, Weight: []float64{0, }},
		//{Bias: .5, Weight: []float64{0, }},
		//{Bias: .5, Weight: []float64{0, }},
	}
	n, err := nn.NewNet(layers, layers)
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
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

func TestArrTwoLayerHeight2(t *testing.T) {
	layers := []nn.LayerSpec{
		{Bias: .5, Weight: []float64{0, 0}},
		{Bias: .5, Weight: []float64{0, 0}},
		//{Bias: .5, Weight: []float64{0, }},
		//{Bias: .5, Weight: []float64{0, }},
	}
	n, err := nn.NewNet(layers, layers)
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
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

func TestArrTwoLayerHeight4(t *testing.T) {
	layers := []nn.LayerSpec{
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
	}
	n, err := nn.NewNet(layers, layers)
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
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

func TestArrThreeHt4(t *testing.T) {
	layers := []nn.LayerSpec{
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
	}
	n, err := nn.NewNet(layers, layers, layers)
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
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

func TestArrFourHt4(t *testing.T) {
	layers := []nn.LayerSpec{
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
	}
	n, err := nn.NewNet(layers, layers, layers, layers)
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
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
func TestArrBack(t *testing.T) {
	layers := []nn.LayerSpec{
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
		{Bias: .5, Weight: []float64{0, 0, 0, 0}},
	}
	n, err := nn.NewNet(layers, layers, layers, layers)
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
	}
	n.Rand(false)
}

// 3-3-1 xor test
func TestArrXOR(t *testing.T) {
	left := []nn.LayerSpec{
		{Bias: .0, Weight: []float64{.351, 1.076, 1.116,},},
		{Bias: .0, Weight: []float64{-.097, -.165, .542},},
		{Bias: .0, Weight: []float64{.457, -.165, -.331},},
	}
	hidden := []nn.LayerSpec{
		{Bias: .0, Weight: []float64{.383, -.327, -.329},},
	}
	n, err := nn.NewNet(left, hidden)
	if err != nil {
		t.Fatalf("NewLayer: got %v, want nil", err)
	}
	//nn.V = t.Logf
	n.Run()

	for i, tt := range []struct {
		in [3]float64
		low bool
	}{
		{in: [3]float64{0,0,1}, low: true,},
		{in: [3]float64{0,1,1}, low: false, },
		{in: [3]float64{1,0,1}, low: false,},
		{in: [3]float64{1,1,1}, low: true,},
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
		t.Logf("%d: of %v", i, of)
		v := of[0]

		t.Logf("%d: got %v, want %v", i,  v, tt.low)
		if v< .5 && tt.low {
			continue
		}
		if v > .5 && ! tt.low {
			continue
		}
		t.Errorf("%d: got %v, want low %v", i, of[0], tt.low)

	}
}
