package nn

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
)

var V = func(string, ...any) {}

// N implements io.WriterAt and io.ReaderAt.
// The offset is the chan number.
type N struct {
	id   string
	in   []chan float64
	wt   []float64
	bias float64
	out  []chan float64
}

func New(id string, out int, bias float64, wt []float64) (*N, error) {
	chans := make([]chan float64, len(wt)+out, len(wt)+out)
	for i := range chans {
		chans[i] = make(chan float64)
	}
	return &N{id: id, in: chans[:len(wt)], wt: wt, bias: bias, out: chans[len(wt):]}, nil
}

func (n *N) String() string {
	return fmt.Sprintf("N%s", n.id)
}

// Rand initializes an n. zeroBias determines
// if the bias should be zero. This will happen
// if we want to just add a bias layer,
// which simplifies life a good deal.
func (n *N) Rand(zeroBias bool) {
	for i := range n.wt {
		n.wt[i] = rand.NormFloat64()
	}
	if zeroBias {
		return
	}
	n.bias = rand.NormFloat64()
}

// Run runs an N
func (n *N) Run() {
	go func() {
		for {
			var x float64
			for i := range n.in {
				V("  %s:Run:recv %d:", n, i)
				f := <-n.in[i]
				V("  %s:Run:got %v wt %v prod %v", n, f, n.wt[i], f*n.wt[i])
				x += f * n.wt[i]
			}
			z := (x - n.bias)
			V("  %s:dot is %v, z is %v", n, x, z)
			y := 1 / (1 + math.Exp(-z))

			V("  %s:%v + %v is %v", n, x, n.bias, y)
			for i := range n.out {
				V("  %s:Run:send %d %v", n, i, y)
				n.out[i] <- y
				V("  %s:Run:Sent", n)
			}
		}
	}()
}

func (n *N) Recv(i uint) float64 {
	// it's just our stuff, so don't be bad.
	if i > uint(len(n.out)) {
		log.Panicf("  %s:Recv %d: only %d chans", n, i, len(n.out))
	}
	x := <-n.out[i]
	V("  %s:recv on %d %v", n, i, x)
	return x
}

func (n *N) Send(i uint, f float64) {
	if i > uint(len(n.in)) {
		log.Panicf("  %s:Send %d: only %d chans", n, i, len(n.in))
	}
	V("  %s:send: %v to %d", n, f, i)
	n.in[i] <- f
}

// Layer is a layer of N
type Layer struct {
	ID string
	NN []*N
}

func (c *Layer) String() string {
	return fmt.Sprintf("C%s", c.ID)
}

type LayerSpec struct {
	Weight []float64
	Bias   float64
}

func NewLayer(id string, layerspec ...LayerSpec) (*Layer, error) {
	layer := &Layer{ID: id, NN: make([]*N, len(layerspec), len(layerspec))}
	for i, c := range layerspec {
		var err error
		if layer.NN[i], err = New(fmt.Sprintf("(%s,%d)", id, i), 1, c.Bias, c.Weight); err != nil {
			return nil, err
		}
	}
	return layer, nil
}

func (c *Layer) Run() {
	for _, n := range c.NN {
		n.Run()
	}
}

func (c *Layer) Rand(zeroBias bool) {
	for i := range c.NN {
		c.NN[i].Rand(zeroBias )
	}
}

func (c *Layer) Recv() []float64 {
	// it's just our stuff, so don't be bad.
	r := make([]float64, len(c.NN), len(c.NN))
	for i := range r {
		V(" %s:call recv from %v", c, c.NN[i])
		r[i] = c.NN[i].Recv(0)
		V(" %s:got %v", c, r[i])
	}
	return r
}

func (c *Layer) Send(i uint, f float64) {
	V(" %s:send %v to %d", c, f, i)
	for j := range c.NN {
		V(" %s:send %v to %d:%d", c, f, j, i)
		go c.NN[j].Send(uint(i), f)
	}
}

// Net is an array of Layer
type Net struct {
	Layers []*Layer
}

func NewNet(layers ...[]LayerSpec) (*Net, error) {
	n := &Net{Layers: make([]*Layer, len(layers), len(layers))}
	for i, layer := range layers {
		var err error
		n.Layers[i], err = NewLayer(fmt.Sprintf("%d", i), layer...)
		if err != nil {
			return nil, err
		}
	}
	// Some simple checking.
	for i, layer := range n.Layers[:len(n.Layers)-1] {
		for j, c := range n.Layers[i+1].NN {
			if len(layer.NN) != len(c.in) {
				return nil, fmt.Errorf("Layer %d: layer[%d].NN[%d] only has %d inputs, need %d:%w", i, i+1, j, len(c.in), len(layer.NN), os.ErrInvalid)
			}
		}

	}
	return n, nil
}

func (n*Net) Rand(zeroBias bool) {
	for  _, layer := range n.Layers {
		layer.Rand(zeroBias)
	}
}

func (n *Net) Run() {
	// For each layer after the first, set up the goroutines to copy one to the next.
	// This can be pretty simple-minded, blocking, in-order movement.
	for _, c := range n.Layers {
		c.Run()
	}

	for i := range n.Layers {
		if i == 0 {
			continue
		}
		V("Set up copiers for %d..%d", i-1, i)
		// each layer has nets of the same size, for now, although the design
		// allows it to vary, let's not go there yet. Until we need to.
		// Each node has one and only one output (for now); fanout is handled
		// by this goroutine.
		// So foreach layer[i].nn[j] goes to every one of layer[i+1].nn[i]
		go func(i int) {
			x := n.Layers[i]
			r := n.Layers[i-1]
			for {
				var pass int
				V("net:Loop for layer %d", i)
				f := r.Recv()
				V("net:recv'd %v on layer %d", f, i)
				for j, v := range f {
					V("net; send %v from layer %d to %d port %d", f, i, i+1, j)
					x.Send(uint(j), v)
				}
				V("net:Loop for layer %d pass %d", i, pass)
			}
		}(i)
	}
}

func (n *Net) Recv() []float64 {
	// We receive from the last layer
	return n.Layers[len(n.Layers)-1].Recv()
}

func (n *Net) Send(i uint, f float64) {
	first := n.Layers[0]
	V("net:send %v to %d", f, i)
	first.Send(uint(i), f)
}

// NewNetEmpty returns a new net with layers and weights
// and connections ready to go.
// It is assumed that it will be
// randomized.
func NewNetEmpty(layers ...uint) (*Net, error) {
	return nil, nil
}
