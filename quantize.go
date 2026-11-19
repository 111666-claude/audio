package audio

import (
	"errors"
	"math"
)

var (
	// ErrUnsupportedBitDepth is returned when a quantization target bit depth
	// other than 8, 16 or 24 is requested.
	ErrUnsupportedBitDepth = errors.New("audio: unsupported bit depth, want 8, 16 or 24")
	// ErrInvalidChannels is returned when a quantizer is created with a
	// non-positive number of channels.
	ErrInvalidChannels = errors.New("audio: channel count must be positive")
	// ErrShortBuffer is returned when the destination slice passed to a
	// Quantizer is too small to hold the source samples.
	ErrShortBuffer = errors.New("audio: destination buffer is too short")
	// ErrQuantizeOverflow is returned if a quantized sample cannot be
	// represented by an int. Supported bit depths (8, 16 and 24) never
	// trigger this; the check is defensive.
	ErrQuantizeOverflow = errors.New("audio: quantized sample overflows int")
)

// QuantizeOptions configures the conversion of normalized float samples into
// signed integer PCM of a lower bit depth.
//
// Dither adds triangular (TPDF) noise with a peak-to-peak amplitude of 2 LSB
// before rounding, which decorrelates the quantization error from the signal.
// ErrorFeedback enables a first-order error feedback loop, which high-passes
// the quantization noise. Both can be enabled at the same time.
//
// Seed makes the process reproducible: the same seed and the same input always
// produce the same output. Each channel gets its own independent random stream
// and feedback accumulator, derived from Seed and the channel index. Seed is
// ignored when Dither is false (no randomness is consumed at all).
type QuantizeOptions struct {
	// BitDepth is the target integer bit depth: 8, 16 or 24.
	BitDepth int
	// Dither adds TPDF dither before quantization.
	Dither bool
	// ErrorFeedback enables first-order error feedback (noise shaping).
	ErrorFeedback bool
	// Seed initializes the per-channel dither streams.
	Seed int64
}

// ValidBitDepth reports whether bits is a supported integer PCM bit depth.
func ValidBitDepth(bits int) bool {
	switch bits {
	case 8, 16, 24:
		return true
	}
	return false
}

// splitMix64 is a small, fast, deterministic PRNG. Separate instances are used
// per channel so channel state can never leak into another channel.
type splitMix64 struct {
	state uint64
}

func (s *splitMix64) next() uint64 {
	s.state += 0x9E3779B97F4A7C15
	z := s.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// uniform returns a uniformly distributed value in [0, 1).
func (s *splitMix64) uniform() float64 {
	return float64(s.next()>>11) / float64(uint64(1)<<53)
}

// Quantizer converts normalized float samples to signed integer PCM.
//
// A Quantizer holds all state needed for block-by-block processing: the dither
// streams and the error-feedback accumulators are kept per channel, and the
// position within the current frame is tracked across calls. Processing a
// buffer in chunks therefore produces exactly the same samples as processing
// it in one call, even if a chunk boundary splits a frame.
//
// A Quantizer is not safe for concurrent use.
type Quantizer struct {
	bitDepth      int
	scale         int64 // 2^(bitDepth-1); also the magnitude of the minimum value
	maxValue      int64 // scale-1, the maximum positive value
	seed          int64
	dither        bool
	errorFeedback bool
	channels      int
	rng           []splitMix64
	feedback      []float64
	pos           int
}

// NewQuantizer creates a quantizer for the given number of channels. It
// returns ErrInvalidChannels if numChannels is not positive and
// ErrUnsupportedBitDepth if opts.BitDepth is not 8, 16 or 24.
func NewQuantizer(numChannels int, opts QuantizeOptions) (*Quantizer, error) {
	if numChannels <= 0 {
		return nil, ErrInvalidChannels
	}
	if !ValidBitDepth(opts.BitDepth) {
		return nil, ErrUnsupportedBitDepth
	}

	q := &Quantizer{
		bitDepth:      opts.BitDepth,
		scale:         int64(1) << uint(opts.BitDepth-1),
		seed:          opts.Seed,
		dither:        opts.Dither,
		errorFeedback: opts.ErrorFeedback,
		channels:      numChannels,
		rng:           make([]splitMix64, numChannels),
		feedback:      make([]float64, numChannels),
	}
	q.maxValue = q.scale - 1
	q.Reset()
	return q, nil
}

// BitDepth returns the integer bit depth this quantizer produces.
func (q *Quantizer) BitDepth() int {
	if q == nil {
		return 0
	}
	return q.bitDepth
}

// NumChannels returns the number of channels this quantizer was created for.
func (q *Quantizer) NumChannels() int {
	if q == nil {
		return 0
	}
	return q.channels
}

// Reset restores the initial state: dither streams go back to the configured
// seed, error-feedback accumulators are cleared and the frame position is
// reset. Processing the same samples again after Reset reproduces the same
// output.
func (q *Quantizer) Reset() {
	if q == nil {
		return
	}
	for i := range q.rng {
		// Each channel starts at a different point in the splitmix64
		// sequence. Adding (i+1) golden-ratio steps keeps the streams
		// independent for every channel.
		q.rng[i] = splitMix64{state: uint64(q.seed) + uint64(i+1)*0x9E3779B97F4A7C15}
	}
	for i := range q.feedback {
		q.feedback[i] = 0
	}
	q.pos = 0
}

// QuantizeFloat32 converts normalized float32 samples into signed integers of
// the quantizer's bit depth.
//
// Input values are expected to be in the [-1, 1] range. Values outside that
// range are clipped to full scale. Non-finite values are mapped explicitly:
// NaN becomes 0, +Inf becomes positive full scale and -Inf becomes negative
// full scale. Samples are never silently wrapped.
//
// If dst is nil a new slice is allocated; otherwise dst must be at least as
// long as src, and only dst[:len(src)] is written. The slice that was written
// is returned. The source slice is never modified.
func (q *Quantizer) QuantizeFloat32(dst []int, src []float32) ([]int, error) {
	if q == nil {
		return nil, ErrInvalidBuffer
	}
	if dst == nil {
		dst = make([]int, len(src))
	} else if len(dst) < len(src) {
		return nil, ErrShortBuffer
	}
	for i, s := range src {
		v, err := q.quantizeOne(float64(s), q.channelAt(i))
		if err != nil {
			return nil, err
		}
		dst[i] = int(v)
	}
	q.advance(len(src))
	return dst[:len(src)], nil
}

// QuantizeFloat64 is the float64 counterpart of QuantizeFloat32.
func (q *Quantizer) QuantizeFloat64(dst []int, src []float64) ([]int, error) {
	if q == nil {
		return nil, ErrInvalidBuffer
	}
	if dst == nil {
		dst = make([]int, len(src))
	} else if len(dst) < len(src) {
		return nil, ErrShortBuffer
	}
	for i, s := range src {
		v, err := q.quantizeOne(s, q.channelAt(i))
		if err != nil {
			return nil, err
		}
		dst[i] = int(v)
	}
	q.advance(len(src))
	return dst[:len(src)], nil
}

func (q *Quantizer) channelAt(i int) int {
	return (q.pos + i) % q.channels
}

func (q *Quantizer) advance(n int) {
	q.pos = (q.pos + n) % q.channels
}

func (q *Quantizer) quantizeOne(sample float64, channel int) (int64, error) {
	// Non-finite values have a defined mapping rather than producing
	// implementation-specific integer conversions.
	switch {
	case math.IsNaN(sample):
		sample = 0
	case math.IsInf(sample, 1):
		sample = 1
	case math.IsInf(sample, -1):
		sample = -1
	}

	// Clip to the normalized full-scale range. This also prevents an
	// unclipped +1.0 from overflowing the positive integer range.
	if sample > 1 {
		sample = 1
	} else if sample < -1 {
		sample = -1
	}

	// Work in LSB units.
	v := sample * float64(q.scale)
	if q.errorFeedback {
		v += q.feedback[channel]
	}
	if q.dither {
		// TPDF dither: difference of two independent uniform samples.
		// The result is in (-1, 1) LSB, i.e. 2 LSB peak to peak.
		v += q.rng[channel].uniform() - q.rng[channel].uniform()
	}

	v = roundHalfAwayFromZero(v)

	// Clip to the symmetric signed integer range. Note that the most
	// negative value is reachable while the positive extreme saturates one
	// step lower, as is usual for two's complement PCM.
	if v < float64(-q.scale) {
		v = float64(-q.scale)
	} else if v > float64(q.maxValue) {
		v = float64(q.maxValue)
	}

	iv := int64(v)
	if int64(int(iv)) != iv {
		return 0, ErrQuantizeOverflow
	}

	if q.errorFeedback {
		// Track the quantization error of the un-dithered value, so the
		// shaping loop does not amplify the dither noise injected above.
		// When a sample is clipped the excess is discarded and the
		// accumulator is bounded so a sustained over-full-scale signal
		// cannot make it grow without limit.
		e := (sample*float64(q.scale) + q.feedback[channel]) - float64(iv)
		if e > 2 {
			e = 2
		} else if e < -2 {
			e = -2
		}
		q.feedback[channel] = e
	}

	return iv, nil
}

// roundHalfAwayFromZero rounds to the nearest integer, with exact halves going
// away from zero. It is spelled out rather than using math.Round to keep the
// package usable with older Go releases.
func roundHalfAwayFromZero(v float64) float64 {
	if v < 0 {
		return math.Ceil(v - 0.5)
	}
	return math.Floor(v + 0.5)
}

// AsIntBufferQuantized converts a normalized Float32Buffer into an IntBuffer
// of opts.BitDepth, optionally applying TPDF dither and first-order error
// feedback. The source buffer is not modified. The returned buffer keeps the
// source sample rate and channel count and records the target bit depth in
// SourceBitDepth.
//
// All samples are converted, including a trailing partial frame if
// len(Data) is not a multiple of NumChannels; the sample count is preserved.
func (buf *Float32Buffer) AsIntBufferQuantized(opts QuantizeOptions) (*IntBuffer, error) {
	if buf == nil || buf.Format == nil {
		return nil, ErrInvalidBuffer
	}
	q, err := NewQuantizer(buf.Format.NumChannels, opts)
	if err != nil {
		return nil, err
	}
	data, err := q.QuantizeFloat32(nil, buf.Data)
	if err != nil {
		return nil, err
	}
	return &IntBuffer{
		Format: &Format{
			NumChannels: buf.Format.NumChannels,
			SampleRate:  buf.Format.SampleRate,
		},
		Data:           data,
		SourceBitDepth: opts.BitDepth,
	}, nil
}

// AsIntBufferQuantized converts a normalized FloatBuffer into an IntBuffer of
// opts.BitDepth, optionally applying TPDF dither and first-order error
// feedback. The source buffer is not modified. The returned buffer keeps the
// source sample rate and channel count and records the target bit depth in
// SourceBitDepth.
//
// All samples are converted, including a trailing partial frame if
// len(Data) is not a multiple of NumChannels; the sample count is preserved.
func (buf *FloatBuffer) AsIntBufferQuantized(opts QuantizeOptions) (*IntBuffer, error) {
	if buf == nil || buf.Format == nil {
		return nil, ErrInvalidBuffer
	}
	q, err := NewQuantizer(buf.Format.NumChannels, opts)
	if err != nil {
		return nil, err
	}
	data, err := q.QuantizeFloat64(nil, buf.Data)
	if err != nil {
		return nil, err
	}
	return &IntBuffer{
		Format: &Format{
			NumChannels: buf.Format.NumChannels,
			SampleRate:  buf.Format.SampleRate,
		},
		Data:           data,
		SourceBitDepth: opts.BitDepth,
	}, nil
}

// QuantizeFloat32Buffer is a convenience wrapper around
// Float32Buffer.AsIntBufferQuantized.
func QuantizeFloat32Buffer(buf *Float32Buffer, opts QuantizeOptions) (*IntBuffer, error) {
	return buf.AsIntBufferQuantized(opts)
}

// QuantizeFloatBuffer is a convenience wrapper around
// FloatBuffer.AsIntBufferQuantized.
func QuantizeFloatBuffer(buf *FloatBuffer, opts QuantizeOptions) (*IntBuffer, error) {
	return buf.AsIntBufferQuantized(opts)
}
