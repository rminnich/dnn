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

func New(wt []float64, bias float64, out int) (*N, error) {
	chans := make([]chan float64, len(wt)+out, len(wt)+out)
	for i := range chans {
		chans[i] = make(chan float64)
	}
	return &N{in: chans[:len(wt)], wt: wt, bias: bias, out: chans[len(wt):]}, nil
}

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
