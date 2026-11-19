package audio

import (
	"errors"
	"math"
)

// DC blocking related errors.
var (
	// ErrInvalidSampleRate is returned when the sample rate is not a
	// strictly positive integer.
	ErrInvalidSampleRate = errors.New("invalid sample rate: must be a strictly positive integer")
	// ErrInvalidCutoff is returned when the cutoff frequency is not strictly
	// positive or when it is not strictly below the Nyquist frequency
	// (sample rate / 2).
	ErrInvalidCutoff = errors.New("invalid cutoff frequency: must be > 0 and < sample rate/2")
	// ErrNonFiniteSample is returned when a float buffer contains a NaN or
	// an Inf sample. The DC blocker state is left untouched in that case.
	ErrNonFiniteSample = errors.New("non-finite sample: NaN or Inf")
	// ErrChannelMismatch is returned when a processed buffer does not have
	// the same number of channels as the DC blocker.
	ErrChannelMismatch = errors.New("channel count mismatch between the buffer and the DC blocker")
	// ErrSampleRateMismatch is returned when a processed buffer does not
	// have the same sample rate as the DC blocker.
	ErrSampleRateMismatch = errors.New("sample rate mismatch between the buffer and the DC blocker")
)

// DCBlocker is a streaming one-pole high-pass filter that removes the DC
// component (constant offset) of a signal while preserving its low frequency
// content above a configurable cutoff frequency.
//
// The filter is a one-pole analog high-pass discretized with the bilinear
// transform (frequency pre-warped so the -3dB point is exactly at fc):
//
//	w0   = tan(pi * fc / fs)
//	g    = 1 / (1 + w0)
//	a1   = (1 - w0) / (1 + w0), strictly in (-1, 1)
//
//	y[n] = g * (x[n] - x[n-1]) + a1 * y[n-1]
//
// It has a unit gain at the Nyquist frequency, a strictly zero gain at DC
// and its -3dB point is exactly at the configured cutoff fc.
//
// A DCBlocker keeps state between calls: it supports chunked/streamed input.
// Feeding a whole buffer at once or feeding it in any number of chunks
// produces exactly the same per-sample output. Each channel keeps an
// independent state. Reset restores the blocker to its initial state.
type DCBlocker struct {
	// Format describes the number of channels and the sample rate the
	// blocker is configured for. Every processed buffer must match it.
	Format *Format
	// cutoff is the -3dB cutoff frequency in Hz.
	cutoff float64
	// gain is the feed-forward coefficient g = 1/(1+w0).
	gain float64
	// a1 is the feedback/pole coefficient, always strictly in (-1, 1).
	a1 float64
	// xPrev holds the last input sample of each channel.
	xPrev []float64
	// yPrev holds the last output sample of each channel.
	yPrev []float64
	// carry holds the trailing samples of an input that did not end on a
	// frame boundary; they are prepended to the next input chunk.
	carry []float64
}

// NewDCBlocker creates a streaming DC blocking filter for the given format
// (number of channels and sample rate) with the given -3dB cutoff frequency
// in Hz.
//
// It returns an error if the format is nil, has no channel, if the sample
// rate is not strictly positive, or if the cutoff is not strictly positive
// and strictly below the Nyquist frequency (sample rate / 2).
func NewDCBlocker(cutoff float64, f *Format) (*DCBlocker, error) {
	if f == nil || f.NumChannels <= 0 {
		return nil, ErrInvalidBuffer
	}
	if f.SampleRate <= 0 {
		return nil, ErrInvalidSampleRate
	}
	if !validCutoff(cutoff, f.SampleRate) {
		return nil, ErrInvalidCutoff
	}
	d := &DCBlocker{
		Format: &Format{
			NumChannels: f.NumChannels,
			SampleRate:  f.SampleRate,
		},
		cutoff: cutoff,
		gain:   dcGain(cutoff, f.SampleRate),
		a1:     dcCoefficient(cutoff, f.SampleRate),
		xPrev:  make([]float64, f.NumChannels),
		yPrev:  make([]float64, f.NumChannels),
	}
	return d, nil
}

// Reset restores the DC blocker to its initial state: the per-channel filter
// memory and any buffered incomplete frame are cleared. After Reset, the
// blocker behaves like a newly created one.
func (d *DCBlocker) Reset() {
	if d == nil {
		return
	}
	for i := range d.xPrev {
		d.xPrev[i] = 0
	}
	for i := range d.yPrev {
		d.yPrev[i] = 0
	}
	d.carry = d.carry[:0]
}

// Process applies the DC blocking filter to a buffer and returns a new
// buffer of the same concrete type, channel count and sample rate. The input
// buffer is never modified.
//
// Input chunks do not need to end on a frame boundary: trailing samples of an
// incomplete frame are buffered and prepended to the next chunk, so arbitrary
// chunk boundaries yield exactly the same result as processing the whole
// signal at once. Note that, by definition, samples of an incomplete frame
// cannot be returned until the frame is completed by a subsequent chunk.
//
// An error is returned (and the state is left unchanged) if the buffer or its
// format is nil, if its channel count or sample rate does not match the
// blocker, or if a float buffer contains a non-finite (NaN/Inf) sample.
func (d *DCBlocker) Process(buf Buffer) (Buffer, error) {
	switch b := buf.(type) {
	case *FloatBuffer:
		return d.ProcessFloatBuffer(b)
	case *Float32Buffer:
		return d.ProcessFloat32Buffer(b)
	case *IntBuffer:
		return d.ProcessIntBuffer(b)
	case *PCMBuffer:
		if b == nil {
			return nil, ErrInvalidBuffer
		}
		return d.ProcessFloatBuffer(b.AsFloatBuffer())
	case nil:
		return nil, ErrInvalidBuffer
	default:
		return nil, ErrInvalidBuffer
	}
}

// ProcessFloatBuffer applies the DC blocking filter to a float64 buffer and
// returns a new FloatBuffer with the same format. See Process for the
// streaming semantics and error conditions.
func (d *DCBlocker) ProcessFloatBuffer(buf *FloatBuffer) (*FloatBuffer, error) {
	if buf == nil {
		return nil, ErrInvalidBuffer
	}
	if err := d.validate(buf); err != nil {
		return nil, err
	}
	for _, v := range buf.Data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, ErrNonFiniteSample
		}
	}
	nCh := d.Format.NumChannels
	samples, _ := d.combine(buf.Data, nCh)
	out := d.filter(samples, nCh)

	return &FloatBuffer{
		Format: d.outputFormat(),
		Data:   out,
	}, nil
}

// ProcessFloat32Buffer applies the DC blocking filter to a float32 buffer and
// returns a new Float32Buffer with the same format. Filtering is performed in
// float64 precision and the result is converted back to float32. See Process
// for the streaming semantics and error conditions.
func (d *DCBlocker) ProcessFloat32Buffer(buf *Float32Buffer) (*Float32Buffer, error) {
	if buf == nil {
		return nil, ErrInvalidBuffer
	}
	if err := d.validate(buf); err != nil {
		return nil, err
	}
	for _, v := range buf.Data {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, ErrNonFiniteSample
		}
	}
	nCh := d.Format.NumChannels
	converted := make([]float64, len(buf.Data))
	for i, v := range buf.Data {
		converted[i] = float64(v)
	}
	combined := d.withCarry(converted)
	samples, _ := d.storeCarry(combined, nCh)
	filtered := d.filter(samples, nCh)

	out := make([]float32, len(filtered))
	for i, v := range filtered {
		out[i] = float32(v)
	}
	return &Float32Buffer{
		Format:         d.outputFormat(),
		Data:           out,
		SourceBitDepth: buf.SourceBitDepth,
	}, nil
}

// ProcessIntBuffer applies the DC blocking filter to an integer buffer and
// returns a new IntBuffer with the same format and SourceBitDepth.
//
// Integer samples are normalized to floats using the source bit depth
// (inferred from the data when SourceBitDepth is 0, like AsFloat32Buffer),
// filtered in float64 and converted back. Conversion rounds half away from
// zero and saturates (clamps) to the representable range of the source bit
// depth instead of wrapping. See Process for the streaming semantics and the
// other error conditions.
func (d *DCBlocker) ProcessIntBuffer(buf *IntBuffer) (*IntBuffer, error) {
	if buf == nil {
		return nil, ErrInvalidBuffer
	}
	if err := d.validate(buf); err != nil {
		return nil, err
	}
	nCh := d.Format.NumChannels
	bitDepth := intBufferBitDepth(buf)
	factor := math.Pow(2, float64(bitDepth)-1)

	normalized := make([]float64, len(buf.Data))
	for i, v := range buf.Data {
		normalized[i] = float64(v) / factor
	}
	combined := d.withCarry(normalized)
	samples, _ := d.storeCarry(combined, nCh)
	filtered := d.filter(samples, nCh)

	out := make([]int, len(filtered))
	for i, v := range filtered {
		out[i] = saturatedInt(roundFloat(v*factor), bitDepth)
	}
	return &IntBuffer{
		Format:         d.outputFormat(),
		Data:           out,
		SourceBitDepth: buf.SourceBitDepth,
	}, nil
}

// validate checks that a buffer can be processed by this blocker. It does not
// mutate the blocker state.
func (d *DCBlocker) validate(buf Buffer) error {
	if d == nil || buf == nil {
		return ErrInvalidBuffer
	}
	f := buf.PCMFormat()
	if f == nil {
		return ErrInvalidBuffer
	}
	if f.NumChannels != d.Format.NumChannels {
		return ErrChannelMismatch
	}
	if f.SampleRate != d.Format.SampleRate {
		return ErrSampleRateMismatch
	}
	return nil
}

// outputFormat returns an independent Format copy for an output buffer.
func (d *DCBlocker) outputFormat() *Format {
	return &Format{
		NumChannels: d.Format.NumChannels,
		SampleRate:  d.Format.SampleRate,
	}
}

// withCarry prepends the buffered incomplete frame to the incoming samples
// and returns the concatenation. The caller then hands the result to
// storeCarry so the new incomplete frame (if any) is buffered again.
func (d *DCBlocker) withCarry(in []float64) []float64 {
	combined := make([]float64, len(d.carry)+len(in))
	copy(combined, d.carry)
	copy(combined[len(d.carry):], in)
	return combined
}

// combine appends the input samples to the buffered incomplete frame and
// stores the new trailing incomplete frame back in the blocker. It returns the
// complete-frames prefix ready to be filtered.
func (d *DCBlocker) combine(in []float64, nCh int) (samples []float64, frames int) {
	return d.storeCarry(d.withCarry(in), nCh)
}

// storeCarry keeps any trailing samples that do not form a complete frame as
// the new carry and returns the complete-frames prefix and its frame count.
func (d *DCBlocker) storeCarry(combined []float64, nCh int) (samples []float64, frames int) {
	frames = len(combined) / nCh
	consumed := frames * nCh
	if leftover := len(combined) - consumed; leftover > 0 {
		// Allocate a fresh slice: reusing d.carry[:0] would have the
		// length 0 even when its capacity is large enough.
		d.carry = append([]float64(nil), combined[consumed:]...)
	} else {
		d.carry = d.carry[:0]
	}
	return combined[:consumed], frames
}

// filter runs the per-channel one-pole high-pass over the interleaved
// complete-frame samples. The per-channel state (xPrev/yPrev) is updated in
// place so consecutive calls stay continuous.
func (d *DCBlocker) filter(samples []float64, nCh int) []float64 {
	out := make([]float64, len(samples))
	for i := 0; i < len(samples); i++ {
		ch := i % nCh
		x := samples[i]
		// y[n] = g*(x[n] - x[n-1]) + a1*y[n-1]
		y := d.gain*(x-d.xPrev[ch]) + d.a1*d.yPrev[ch]
		out[i] = y
		d.xPrev[ch] = x
		d.yPrev[ch] = y
	}
	return out
}

// validCutoff reports whether cutoff is a finite, strictly positive frequency
// strictly below the Nyquist frequency.
func validCutoff(cutoff float64, sampleRate int) bool {
	if math.IsNaN(cutoff) || math.IsInf(cutoff, 0) || cutoff <= 0 {
		return false
	}
	return cutoff < float64(sampleRate)/2
}

// dcCoefficient computes the pole coefficient a1 of the bilinear-transform
// one-pole high-pass for the given cutoff and sample rate. The result is
// always strictly in (-1, 1) when validCutoff passes.
func dcCoefficient(cutoff float64, sampleRate int) float64 {
	w0 := math.Tan(math.Pi * cutoff / float64(sampleRate))
	return (1 - w0) / (1 + w0)
}

// dcGain computes the feed-forward coefficient g of the bilinear-transform
// one-pole high-pass. g and a1 share the same (1+w0) denominator so that the
// filter gain at the Nyquist frequency is exactly 1.
func dcGain(cutoff float64, sampleRate int) float64 {
	w0 := math.Tan(math.Pi * cutoff / float64(sampleRate))
	return 1 / (1 + w0)
}

// intBufferBitDepth returns the source bit depth of an IntBuffer, inferring
// it from the data the same way IntBuffer.AsFloat32Buffer does when
// SourceBitDepth is not set. The buffer is not mutated.
func intBufferBitDepth(buf *IntBuffer) int {
	if buf.SourceBitDepth > 0 {
		return buf.SourceBitDepth
	}
	bitDepth := 8
	var max int
	for _, s := range buf.Data {
		if s > max {
			max = s
		}
	}
	if max > 127 {
		bitDepth = 16
	}
	if max > 32767 {
		bitDepth = 24
	}
	if max > 8388607 {
		bitDepth = 32
	}
	if int64(max) > 4294967295 {
		bitDepth = 64
	}
	return bitDepth
}

// roundFloat rounds to the nearest integer, rounding half away from zero.
// math.Round is only available since Go 1.10 and the package still supports
// older versions. The half threshold is lowered by a few ULPs so that values
// whose float64 representation sits a hair below an exact half (e.g. 123.5)
// are still treated as halves.
func roundFloat(v float64) float64 {
	t := math.Trunc(v)
	halfEpsilon := 4 * math.SmallestNonzeroFloat64 * math.Abs(v)
	if d := math.Abs(v - t); d >= 0.5-halfEpsilon {
		return t + math.Copysign(1, v)
	}
	return t
}

// saturatedInt converts a float64 to an int, clamping the result to the
// representable signed integer range of the given bit depth instead of
// wrapping on overflow.
func saturatedInt(v float64, bitDepth int) int {
	if bitDepth >= 64 {
		// 2^63 and 2^63-1 are not both representable exactly in float64,
		// handle the 64 bit case with explicit integer bounds.
		if v >= float64(math.MaxInt64) {
			return math.MaxInt64
		}
		if v <= float64(math.MinInt64) {
			return math.MinInt64
		}
		return int(v)
	}
	limit := math.Pow(2, float64(bitDepth)-1)
	// Round the bounds to exact integer floats so clamping an in-range value
	// does not truncate it one unit below the maximum (limit-1 is not always
	// representable exactly in float64).
	hi := math.Floor(limit - 0.5)
	lo := -limit
	if v > hi {
		v = hi
	}
	if v < lo {
		v = lo
	}
	return int(roundFloat(v))
}
