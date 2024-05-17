package nn

import (
	"log"
	"math"
)

var V = func(string, ...any) {}

// N implements io.WriterAt and io.ReaderAt.
// The offset is the chan number.
type N struct {
	in   []chan float64
	wt   []float64
	bias float64
	out  []chan float64
}

func New(out int, bias float64, wt []float64) (*N, error) {
	chans := make([]chan float64, len(wt)+out, len(wt)+out)
	for i := range chans {
		chans[i] = make(chan float64)
	}
	return &N{in: chans[:len(wt)], wt: wt, bias: bias, out: chans[len(wt):]}, nil
}

// Run runs an N
func (n *N) Run() {
	go func() {
		for {
			var x float64
			for i := range n.in {
				V("Run: Recv %d:", i)
				f := <-n.in[i]
				V("Run:Got %v", f)
				x += f * n.wt[i]
			}
			z := (x - n.bias)
			y := 1 / (1 + math.Exp(-z))

			V("%v + %v is %v", x, n.bias, y)
			for i := range n.out {
				V("Run:Send %d %v", i, y)
				n.out[i] <- y
				V("Run:Sent")
			}
		}
	}()
}

func (n *N) Recv(i uint) float64 {
	// it's just our stuff, so don't be bad.
	if i > uint(len(n.out)) {
		log.Panicf("Recv %d: only %d chans", i, len(n.out))
	}
	x := <-n.out[i]
	V("Recv on %d %v", i, x)
	return x
}

func (n *N) Send(i uint, f float64) {
	if i > uint(len(n.in)) {
		log.Panicf("Send %d: only %d chans", i, len(n.in))
	}
	V("Send %v to %d", f, i)
	n.in[i] <- f
}

// Col is a column of N
type Col struct {
	NN []*N
}

type ColSpec struct {
	Weight []float64
	Bias   float64
}

func NewCol(colspec ...ColSpec) (*Col, error) {
	col := &Col{NN: make([]*N, len(colspec), len(colspec))}
	for i, c := range colspec {
		var err error
		if col.NN[i], err = New(1, c.Bias, c.Weight); err != nil {
			return nil, err
		}
	}
	return col, nil
}

func (c *Col) Run() {
	for _, n := range c.NN {
		n.Run()
	}
}

func (c *Col) Recv() []float64 {
	// it's just our stuff, so don't be bad.
	r := make([]float64, len(c.NN), len(c.NN))
	for i := range r {
		V("recv from %v", c.NN[i])
		r[i] = c.NN[i].Recv(0)
	}
	return r
}

func (c *Col) Send(i uint, f float64) {
	V("Send %v to %d", f, i)
	for j := range c.NN {
		V("Send %v to %d:%d", f, j, i)
		go c.NN[j].Send(uint(i), f)
	}
}

// Net is an array of Col
type Net struct {
	Cols []*Col
}

func NewNet(cols ...[]ColSpec) (*Net, error) {
	n := &Net{Cols: make([]*Col, len(cols), len(cols))}
	for i, col := range cols {
		var err error
		n.Cols[i], err = NewCol(col...)
		if err != nil {
			return nil, err
		}
	}
	return n, nil
}

func (n *Net) Run() {
	// For each column after the first, set up the goroutines to copy one to the next.
	// This can be pretty simple-minded, blocking, in-order movement.
	for _, c := range n.Cols {
		c.Run()
	}

	for i := range n.Cols[:len(n.Cols)-1] {
		V("Set up copiers for 0..%d", len(n.Cols)-1)
		// each column has nets of the same size, for now, although the design
		// allows it to vary, let's not go there yet. Until we need to.
		// Each node has one and only one output (for now); fanout is handled
		// by this goroutine.
		// So foreach col[i].nn[j] goes to every one of col[i+1].nn[i]
		go func() {
			for {
				var pass int
				V("Loop for layer %d", i)
				for i, x := range n.Cols[i].NN {
					f := x.Recv(0)
					V("Copy %v from layer %d to %d", f, i, i+1)
					for j, r := range n.Cols[i+1].NN {
						r.Send(uint(j), f)
					}
				}
				V("Loop for layer %d pass %d", i, pass)
			}
		}()
	}
}

func (n *Net) Recv() []float64 {
	// We receive from the last column
	last := n.Cols[len(n.Cols)-1]
	// it's just our stuff, so don't be bad.
	r := make([]float64, len(last.NN), len(last.NN))
	for i := range r {
		V("recv from %v", last.NN[i])
		r[i] = last.NN[i].Recv(0)
	}
	return r
}

func (n *Net) Send(i uint, f float64) {
	first := n.Cols[0]
	V("Send %v to %d", f, i)
	for j := range first.NN {
		V("Send %v to %d:%d", f, j, i)
		go first.NN[j].Send(uint(i), f)
	}
}
