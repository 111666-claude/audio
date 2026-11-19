package audio

import (
	"math"
	"testing"
)

// --- helpers -----------------------------------------------------------

func testChannelSignal(amplitude, freq float64, n, channels int) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = amplitude * math.Sin(2*math.Pi*freq*float64(i/channels)/44100)
	}
	return out
}

func testFloatBuffer(samples []float64, channels, rate int) *FloatBuffer {
	return &FloatBuffer{
		Format: &Format{NumChannels: channels, SampleRate: rate},
		Data:   samples,
	}
}

func testFloat32Buffer(samples []float64, channels, rate int) *Float32Buffer {
	f := make([]float32, len(samples))
	for i, s := range samples {
		f[i] = float32(s)
	}
	return &Float32Buffer{
		Format:         &Format{NumChannels: channels, SampleRate: rate},
		Data:           f,
		SourceBitDepth: 32,
	}
}

func testQuantizer(t *testing.T, channels int, opts QuantizeOptions) *Quantizer {
	t.Helper()
	q, err := NewQuantizer(channels, opts)
	if err != nil {
		t.Fatalf("NewQuantizer(%d, %+v) failed: %v", channels, opts, err)
	}
	return q
}

func testRange(t *testing.T, data []int, bits int) {
	t.Helper()
	scale := int64(1) << uint(bits-1)
	max := scale - 1
	for i, v := range data {
		if int64(int(v)) != int64(v) || int64(v) < -scale || int64(v) > max {
			t.Fatalf("sample %d = %d out of range [%d, %d] for %d bits", i, v, -scale, max, bits)
		}
	}
}

func testEqualInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func testEqualFloats64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func testEqualFloats32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// --- quantizer construction and validation -----------------------------

func TestNewQuantizerValidation(t *testing.T) {
	for _, bits := range []int{-8, 0, 1, 7, 12, 15, 17, 23, 25, 32, 64} {
		if _, err := NewQuantizer(2, QuantizeOptions{BitDepth: bits}); err != ErrUnsupportedBitDepth {
			t.Errorf("NewQuantizer bit depth %d: got %v, want ErrUnsupportedBitDepth", bits, err)
		}
	}
	for _, ch := range []int{0, -1, -2} {
		if _, err := NewQuantizer(ch, QuantizeOptions{BitDepth: 16}); err != ErrInvalidChannels {
			t.Errorf("NewQuantizer channels %d: got %v, want ErrInvalidChannels", ch, err)
		}
	}
	for _, bits := range []int{8, 16, 24} {
		q, err := NewQuantizer(2, QuantizeOptions{BitDepth: bits})
		if err != nil {
			t.Fatalf("NewQuantizer(%d) failed: %v", bits, err)
		}
		if q.BitDepth() != bits || q.NumChannels() != 2 {
			t.Errorf("quantizer reports %d bits / %d channels, want %d / 2", q.BitDepth(), q.NumChannels(), bits)
		}
	}
	if !ValidBitDepth(8) || !ValidBitDepth(16) || !ValidBitDepth(24) {
		t.Error("ValidBitDepth rejected a supported depth")
	}
	if ValidBitDepth(0) || ValidBitDepth(32) {
		t.Error("ValidBitDepth accepted an unsupported depth")
	}
}

func TestQuantizerNilAndBufferErrors(t *testing.T) {
	var q *Quantizer
	if _, err := q.QuantizeFloat64(nil, []float64{0}); err != ErrInvalidBuffer {
		t.Errorf("nil quantizer: got %v, want ErrInvalidBuffer", err)
	}
	if q.BitDepth() != 0 || q.NumChannels() != 0 {
		t.Error("nil quantizer accessors should return 0")
	}

	var f32 *Float32Buffer
	if _, err := f32.AsIntBufferQuantized(QuantizeOptions{BitDepth: 16}); err != ErrInvalidBuffer {
		t.Errorf("nil Float32Buffer: got %v, want ErrInvalidBuffer", err)
	}
	var f64 *FloatBuffer
	if _, err := f64.AsIntBufferQuantized(QuantizeOptions{BitDepth: 16}); err != ErrInvalidBuffer {
		t.Errorf("nil FloatBuffer: got %v, want ErrInvalidBuffer", err)
	}
	if _, err := QuantizeFloat32Buffer(nil, QuantizeOptions{BitDepth: 16}); err != ErrInvalidBuffer {
		t.Errorf("nil Float32Buffer wrapper: got %v, want ErrInvalidBuffer", err)
	}
	if _, err := QuantizeFloatBuffer(nil, QuantizeOptions{BitDepth: 16}); err != ErrInvalidBuffer {
		t.Errorf("nil FloatBuffer wrapper: got %v, want ErrInvalidBuffer", err)
	}

	buf := testFloat32Buffer([]float64{0, 0}, 0, 44100)
	buf.Format.NumChannels = 0
	if _, err := buf.AsIntBufferQuantized(QuantizeOptions{BitDepth: 16}); err != ErrInvalidChannels {
		t.Errorf("zero channels: got %v, want ErrInvalidChannels", err)
	}
	buf = testFloat32Buffer([]float64{0, 0}, 1, 44100)
	if _, err := buf.AsIntBufferQuantized(QuantizeOptions{BitDepth: 32}); err != ErrUnsupportedBitDepth {
		t.Errorf("bad depth from buffer: got %v, want ErrUnsupportedBitDepth", err)
	}

	q = testQuantizer(t, 1, QuantizeOptions{BitDepth: 16})
	dst := make([]int, 2)
	if _, err := q.QuantizeFloat64(dst, []float64{0, 0, 0}); err != ErrShortBuffer {
		t.Errorf("short dst: got %v, want ErrShortBuffer", err)
	}
}

// --- exact quantization and clipping -----------------------------------

func TestQuantizeExactValues(t *testing.T) {
	for _, bits := range []int{8, 16, 24} {
		q := testQuantizer(t, 1, QuantizeOptions{BitDepth: bits})
		scale := float64(int64(1) << uint(bits-1))
		max := float64(scale - 1)

		in := []float64{-1, -0.5, 0, 0.5, 1}
		got, err := q.QuantizeFloat64(nil, in)
		if err != nil {
			t.Fatalf("%d bits: %v", bits, err)
		}
		want := []int{-int(scale), -int(scale / 2), 0, int(scale / 2), int(max)}
		if !testEqualInts(got, want) {
			t.Errorf("%d bits exact values: got %v, want %v", bits, got, want)
		}
		testRange(t, got, bits)
	}
}

func TestQuantizeClippingAndNonFinite(t *testing.T) {
	q := testQuantizer(t, 1, QuantizeOptions{BitDepth: 16})
	nan := math.NaN()
	inf := math.Inf(1)
	ninf := math.Inf(-1)
	in := []float64{2, -2, 1e9, -1e9, nan, inf, ninf, 1 - 1e-12, -1 + 1e-12}
	got, err := q.QuantizeFloat64(nil, in)
	if err != nil {
		t.Fatal(err)
	}
	scale := 32768
	if got[0] != scale-1 || got[1] != -scale || got[2] != scale-1 || got[3] != -scale {
		t.Errorf("out-of-range samples not clipped: %v", got[:4])
	}
	if got[4] != 0 {
		t.Errorf("NaN mapped to %d, want 0", got[4])
	}
	if got[5] != scale-1 {
		t.Errorf("+Inf mapped to %d, want %d", got[5], scale-1)
	}
	if got[6] != -scale {
		t.Errorf("-Inf mapped to %d, want %d", got[6], -scale)
	}
	testRange(t, got, 16)

	// Same handling for the float32 path.
	f32 := []float32{float32(nan), float32(inf), float32(ninf), 3, -3}
	q = testQuantizer(t, 1, QuantizeOptions{BitDepth: 8})
	got32, err := q.QuantizeFloat32(nil, f32)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{0, 127, -128, 127, -128}; !testEqualInts(got32, want) {
		t.Errorf("float32 non-finite/clipping: got %v, want %v", got32, want)
	}
	testRange(t, got32, 8)
}

func TestQuantizeRangeAllBitDepths(t *testing.T) {
	samples := testChannelSignal(1.05, 440, 2048, 1)
	for _, bits := range []int{8, 16, 24} {
		buf := testFloat64Buffer(samples)
		out, err := QuantizeFloatBuffer(buf, QuantizeOptions{BitDepth: bits, Dither: true, ErrorFeedback: true, Seed: 7})
		if err != nil {
			t.Fatal(err)
		}
		testRange(t, out.Data, bits)
	}
	samples32 := make([]float32, len(samples))
	for i, s := range samples {
		samples32[i] = float32(s)
	}
	for _, bits := range []int{8, 16, 24} {
		out, err := QuantizeFloat32Buffer(testFloat32Buffer(samples, 1, 48000), QuantizeOptions{
			BitDepth: bits, Dither: true, ErrorFeedback: true, Seed: 7,
		})
		if err != nil {
			t.Fatal(err)
		}
		testRange(t, out.Data, bits)
	}
}

func testFloat64Buffer(samples []float64) *FloatBuffer {
	return testFloatBuffer(samples, 1, 44100)
}

// --- buffer metadata, source immutability ------------------------------

func TestQuantizeBufferMetadataAndSourceUntouched(t *testing.T) {
	samples := testChannelSignal(0.25, 1000, 300, 2)
	buf := testFloatBuffer(samples, 2, 96000)
	before := append([]float64(nil), buf.Data...)

	out, err := buf.AsIntBufferQuantized(QuantizeOptions{BitDepth: 24, Dither: true, ErrorFeedback: true, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	if out.PCMFormat().NumChannels != 2 || out.PCMFormat().SampleRate != 96000 {
		t.Errorf("format changed: %+v", out.PCMFormat())
	}
	if out.SourceBitDepth != 24 {
		t.Errorf("SourceBitDepth = %d, want 24", out.SourceBitDepth)
	}
	if len(out.Data) != len(buf.Data) {
		t.Errorf("sample count changed: %d != %d", len(out.Data), len(buf.Data))
	}
	if !testEqualFloats64(buf.Data, before) {
		t.Error("source FloatBuffer was modified")
	}
	if out.NumFrames() != buf.NumFrames() {
		t.Errorf("frame count changed: %d != %d", out.NumFrames(), buf.NumFrames())
	}

	f32 := testFloat32Buffer(samples, 2, 22050)
	before32 := append([]float32(nil), f32.Data...)
	out32, err := f32.AsIntBufferQuantized(QuantizeOptions{BitDepth: 8, Dither: true, Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	if out32.PCMFormat().NumChannels != 2 || out32.PCMFormat().SampleRate != 22050 {
		t.Errorf("float32 format changed: %+v", out32.PCMFormat())
	}
	if out32.SourceBitDepth != 8 || len(out32.Data) != len(f32.Data) {
		t.Errorf("float32 output metadata/size wrong: depth=%d len=%d", out32.SourceBitDepth, len(out32.Data))
	}
	if !testEqualFloats32(f32.Data, before32) {
		t.Error("source Float32Buffer was modified")
	}
}

// --- determinism and seed behavior -------------------------------------

func TestQuantizeDeterministicAndSeedDependent(t *testing.T) {
	samples := testChannelSignal(0.1, 997, 2000, 2)
	opts := QuantizeOptions{BitDepth: 16, Dither: true, ErrorFeedback: true, Seed: 42}

	a, _ := QuantizeFloatBuffer(testFloatBuffer(samples, 2, 44100), opts)
	b, _ := QuantizeFloatBuffer(testFloatBuffer(samples, 2, 44100), opts)
	if !testEqualInts(a.Data, b.Data) {
		t.Error("same seed and input produced different output")
	}

	opts.Seed = 43
	c, _ := QuantizeFloatBuffer(testFloatBuffer(samples, 2, 44100), opts)
	if testEqualInts(a.Data, c.Data) {
		t.Error("different seeds produced identical dithered output")
	}

	// Reset must reproduce the original stream.
	q := testQuantizer(t, 2, QuantizeOptions{BitDepth: 16, Dither: true, ErrorFeedback: true, Seed: 42})
	first, _ := q.QuantizeFloat64(nil, samples)
	q.Reset()
	second, _ := q.QuantizeFloat64(nil, samples)
	if !testEqualInts(first, second) {
		t.Error("Reset did not restore the initial dither/feedback state")
	}

	// Without dither the seed must not matter at all.
	noDither := QuantizeOptions{BitDepth: 16, Seed: 1}
	d, _ := QuantizeFloatBuffer(testFloatBuffer(samples, 2, 44100), noDither)
	noDither.Seed = 999
	e, _ := QuantizeFloatBuffer(testFloatBuffer(samples, 2, 44100), noDither)
	if !testEqualInts(d.Data, e.Data) {
		t.Error("seed affected output when dither is disabled")
	}
}

// --- chunked processing equivalence ------------------------------------

func TestQuantizeChunkedMatchesWhole(t *testing.T) {
	const channels = 2
	samples := testChannelSignal(0.3, 523, 1001, channels) // odd sample count => partial frame
	for _, bits := range []int{8, 16, 24} {
		for _, opts := range []QuantizeOptions{
			{BitDepth: bits},
			{BitDepth: bits, Dither: true, Seed: 11},
			{BitDepth: bits, ErrorFeedback: true},
			{BitDepth: bits, Dither: true, ErrorFeedback: true, Seed: 11},
		} {
			whole := testQuantizer(t, channels, opts)
			want, err := whole.QuantizeFloat64(nil, samples)
			if err != nil {
				t.Fatal(err)
			}

			// Chunk sizes deliberately include 1 (splits every frame),
			// odd sizes that split frames, and sizes larger than the data.
			for _, chunk := range []int{1, 2, 3, 5, 7, 16, 1000, len(samples), len(samples) + 10} {
				q := testQuantizer(t, channels, opts)
				got := make([]int, 0, len(samples))
				for off := 0; off < len(samples); off += chunk {
					end := off + chunk
					if end > len(samples) {
						end = len(samples)
					}
					part, err := q.QuantizeFloat64(nil, samples[off:end])
					if err != nil {
						t.Fatalf("%+v chunk %d: %v", opts, chunk, err)
					}
					got = append(got, part...)
				}
				if !testEqualInts(got, want) {
					t.Fatalf("%+v chunk %d differs from whole-buffer output", opts, chunk)
				}
			}

			// Same for the float32 path.
			f32 := testFloat32Buffer(samples, channels, 44100)
			whole32, err := f32.AsIntBufferQuantized(opts)
			if err != nil {
				t.Fatal(err)
			}
			f32q := testQuantizer(t, channels, opts)
			got32 := make([]int, 0, len(samples))
			for off := 0; off < len(samples); off += 3 {
				end := off + 3
				if end > len(samples) {
					end = len(samples)
				}
				part, err := f32q.QuantizeFloat32(nil, f32.Data[off:end])
				if err != nil {
					t.Fatal(err)
				}
				got32 = append(got32, part...)
			}
			if !testEqualInts(got32, whole32.Data) {
				t.Fatalf("%+v float32 chunked output differs", opts)
			}
		}
	}
}

func TestQuantizeChunkedReusesDestination(t *testing.T) {
	samples := testChannelSignal(0.5, 300, 500, 1)
	opts := QuantizeOptions{BitDepth: 16, Dither: true, ErrorFeedback: true, Seed: 5}
	whole, _ := testQuantizer(t, 1, opts).QuantizeFloat64(nil, samples)

	q := testQuantizer(t, 1, opts)
	dst := make([]int, 64)
	got := make([]int, 0, len(samples))
	for off := 0; off < len(samples); off += 64 {
		end := off + 64
		if end > len(samples) {
			end = len(samples)
		}
		n := end - off
		out, err := q.QuantizeFloat64(dst[:n], samples[off:end])
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != n {
			t.Fatalf("got %d samples, want %d", len(out), n)
		}
		got = append(got, out...)
	}
	if !testEqualInts(got, whole) {
		t.Error("reused destination produced different output")
	}
}

// --- per-channel independence -----------------------------------------

func TestQuantizeChannelsIndependent(t *testing.T) {
	// Both channels carry the same signal. A per-channel random stream
	// means channel 0 of the stereo quantizer consumes exactly the same
	// dither as a mono quantizer with the same seed, so the left channel
	// must match the mono output sample for sample. Channel 1 has its own
	// stream and must therefore differ from channel 0.
	sig := testChannelSignal(0.05, 733, 1200, 1)
	monoData := sig
	stereoData := make([]float64, len(sig)*2)
	for i, s := range sig {
		stereoData[2*i] = s
		stereoData[2*i+1] = s
	}
	opts := QuantizeOptions{BitDepth: 16, Dither: true, ErrorFeedback: true, Seed: 1234}

	mono, err := QuantizeFloatBuffer(testFloatBuffer(monoData, 1, 44100), opts)
	if err != nil {
		t.Fatal(err)
	}
	stereo, err := QuantizeFloatBuffer(testFloatBuffer(stereoData, 2, 44100), opts)
	if err != nil {
		t.Fatal(err)
	}

	left := make([]int, len(sig))
	right := make([]int, len(sig))
	for i := range sig {
		left[i] = stereo.Data[2*i]
		right[i] = stereo.Data[2*i+1]
	}
	if !testEqualInts(left, mono.Data) {
		t.Error("left channel differs from mono output: channels are not independent streams")
	}
	if testEqualInts(left, right) {
		t.Error("left and right channels are identical; per-channel dither state not independent")
	}
}

// --- dither statistics on a low level sine -----------------------------

func quantErrorStats(t *testing.T, out []int, in []float64, bits int) (mean, stddev, min, max float64) {
	t.Helper()
	scale := float64(int64(1) << uint(bits-1))
	if len(out) != len(in) {
		t.Fatalf("length mismatch %d != %d", len(out), len(in))
	}
	var sum, sumSq float64
	min, max = math.Inf(1), math.Inf(-1)
	for i := range in {
		e := float64(out[i])/scale - in[i]
		sum += e
		sumSq += e * e
		if e < min {
			min = e
		}
		if e > max {
			max = e
		}
	}
	n := float64(len(in))
	mean = sum / n
	variance := sumSq/n - mean*mean
	if variance < 0 {
		variance = 0
	}
	return mean, math.Sqrt(variance), min, max
}

func TestQuantizeSilence(t *testing.T) {
	for _, bits := range []int{8, 16, 24} {
		scale := float64(int64(1) << uint(bits-1))
		silence := make([]float64, 4096)
		buf := testFloatBuffer(silence, 1, 44100)

		// Without dither, digital silence must stay exactly zero.
		plain, err := QuantizeFloatBuffer(buf, QuantizeOptions{BitDepth: bits})
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range plain.Data {
			if v != 0 {
				t.Fatalf("%d bits: silence without dither produced %d at %d", bits, v, i)
			}
		}

		// With TPDF dither the output must stay within +/-1 LSB and be
		// centred on zero.
		dith, err := QuantizeFloatBuffer(buf, QuantizeOptions{BitDepth: bits, Dither: true, Seed: 9})
		if err != nil {
			t.Fatal(err)
		}
		var sum int64
		nonzero := 0
		for _, v := range dith.Data {
			if v < -1 || v > 1 {
				t.Fatalf("%d bits: silence dither produced %d, want -1..1 (new = %.3f LSB)", bits, v, float64(v))
			}
			if v != 0 {
				nonzero++
			}
			sum += int64(v)
		}
		if nonzero == 0 {
			t.Errorf("%d bits: dither produced no activity on silence", bits)
		}
		// -1 and +1 should both occur; mean must be tiny.
		mean := float64(sum) / float64(len(dith.Data))
		if math.Abs(mean) > 0.01 {
			t.Errorf("%d bits: dithered silence mean = %f LSB, want ~0", bits, mean)
		}
		if scale < 1 {
			t.Fatal("bad scale")
		}
	}
}

func TestQuantizeLowLevelSine(t *testing.T) {
	const bits = 16
	const n = 48000
	// -60 dBFS-ish: ~32 LSB peak at 16 bits.
	in := testChannelSignal(0.001, 997, n, 1)
	buf := testFloatBuffer(in, 1, 44100)

	plainOpts := QuantizeOptions{BitDepth: bits}
	ditherOpts := QuantizeOptions{BitDepth: bits, Dither: true, Seed: 77}

	plain, err := QuantizeFloatBuffer(buf, plainOpts)
	if err != nil {
		t.Fatal(err)
	}
	dither, err := QuantizeFloatBuffer(buf, ditherOpts)
	if err != nil {
		t.Fatal(err)
	}
	testRange(t, plain.Data, bits)
	testRange(t, dither.Data, bits)

	pMean, pStd, _, _ := quantErrorStats(t, plain.Data, in, bits)
	dMean, dStd, dMin, dMax := quantErrorStats(t, dither.Data, in, bits)

	// Both should be unbiased at DC.
	if math.Abs(pMean) > 0.5 || math.Abs(dMean) > 0.5 {
		t.Errorf("error mean too large: plain=%f LSB dither=%f LSB", pMean, dMean)
	}

	// Error without dither is bounded by 0.5 LSB (rounding only). With
	// TPDF dither it can reach 1.5 LSB but must still be limited.
	if pStd > 0.5 {
		t.Errorf("undithered error stddev = %f LSB, want <= 0.5", pStd)
	}
	if dMin < -1.6 || dMax > 1.6 {
		t.Errorf("dithered error spans [%f, %f] LSB, want within +/-1.5", dMin, dMax)
	}

	// Error is never larger than rounding plus dither plus feedback can
	// explain, for either variant.
	if dStd > 1.0 {
		t.Errorf("dithered error stddev = %f LSB, want <= 1.0", dStd)
	}

	// The dither itself must actually change the low order bits.
	if testEqualInts(plain.Data, dither.Data) {
		t.Error("dither had no effect on a low level signal")
	}
}

func TestQuantizeNearFullScaleStereo(t *testing.T) {
	const bits = 16
	const n = 4000
	left := testChannelSignal(0.999, 440, n, 1)
	right := testChannelSignal(1.02, 660, n, 1) // deliberately over full scale
	data := make([]float64, 2*n)
	for i := 0; i < n; i++ {
		data[2*i] = left[i]
		data[2*i+1] = right[i]
	}
	buf := testFloatBuffer(data, 2, 48000)

	for _, opts := range []QuantizeOptions{
		{BitDepth: bits},
		{BitDepth: bits, Dither: true, Seed: 1},
		{BitDepth: bits, ErrorFeedback: true},
		{BitDepth: bits, Dither: true, ErrorFeedback: true, Seed: 1},
	} {
		out, err := buf.AsIntBufferQuantized(opts)
		if err != nil {
			t.Fatal(err)
		}
		testRange(t, out.Data, bits)
		// Channel assignment must be preserved: the over-driven right
		// channel must hit the positive rail far more often than the left.
		leftClipped, rightClipped := 0, 0
		for i := 0; i < n; i++ {
			if out.Data[2*i] == 32767 {
				leftClipped++
			}
			if out.Data[2*i+1] == 32767 {
				rightClipped++
			}
		}
		if rightClipped == 0 {
			t.Errorf("%+v: over-driven right channel never clipped", opts)
		}
		if rightClipped <= leftClipped {
			t.Errorf("%+v: channel order looks swapped (left clips %d, right clips %d)", opts, leftClipped, rightClipped)
		}
	}
}

// --- no dither, no feedback: plain rounding ---------------------------

func TestQuantizeNoDitherExactRounding(t *testing.T) {
	q := testQuantizer(t, 1, QuantizeOptions{BitDepth: 16})
	// In 16 bit LSB units: +0.25 rounds down, +0.75 rounds up, ties away
	// from zero.
	in := []float64{0.25 / 32768, 0.75 / 32768, 0.5 / 32768, -0.5 / 32768, -0.75 / 32768}
	got, err := q.QuantizeFloat64(nil, in)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{0, 1, 1, -1, -1}; !testEqualInts(got, want) {
		t.Errorf("rounding: got %v, want %v", got, want)
	}
}

// --- error feedback reduces low-band noise on a DC-ish signal ---------

func TestQuantizeErrorFeedbackShapesNoise(t *testing.T) {
	// A constant value just above a quantization step: without feedback the
	// error is a constant DC offset; with first-order feedback it is pushed
	// into alternating high frequency patterns.
	const bits = 16
	q := testQuantizer(t, 1, QuantizeOptions{BitDepth: bits, ErrorFeedback: true})
	step := 1.0 / 32768
	in := make([]float64, 1000)
	for i := range in {
		in[i] = 4.4 * step
	}
	out, err := q.QuantizeFloat64(nil, in)
	if err != nil {
		t.Fatal(err)
	}
	testRange(t, out, bits)
	// The average output must still track the input (no DC introduced by
	// the feedback loop).
	var sum float64
	for _, v := range out {
		sum += float64(v) * step
	}
	mean := sum / float64(len(out))
	if math.Abs(mean-4.4*step) > step {
		t.Errorf("error feedback shifted the DC level: mean=%f input=%f", mean, 4.4*step)
	}
	if _, _, min, max := quantErrorStats(t, out, in, bits); min < -2.1 || max > 2.1 {
		t.Errorf("feedback error out of bounds: [%f, %f] LSB", min, max)
	}
}

// --- trailing partial frame -------------------------------------------

func TestQuantizePartialFrame(t *testing.T) {
	const channels = 2
	// 5 samples = 2 full frames + 1 sample.
	samples := []float64{0.1, -0.1, 0.2, -0.2, 0.3}
	buf := testFloatBuffer(samples, channels, 44100)
	out, err := buf.AsIntBufferQuantized(QuantizeOptions{BitDepth: 16, Dither: true, Seed: 4})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Data) != len(samples) {
		t.Fatalf("partial frame dropped samples: %d != %d", len(out.Data), len(samples))
	}
	if out.NumFrames() != 2 {
		t.Errorf("NumFrames = %d, want 2", out.NumFrames())
	}
	// Chunked conversion of the same data must match sample for sample,
	// including across the broken frame boundary.
	q := testQuantizer(t, channels, QuantizeOptions{BitDepth: 16, Dither: true, Seed: 4})
	var chunked []int
	for _, chunk := range [][]float64{samples[:1], samples[1:4], samples[4:]} {
		part, err := q.QuantizeFloat64(nil, chunk)
		if err != nil {
			t.Fatal(err)
		}
		chunked = append(chunked, part...)
	}
	if !testEqualInts(chunked, out.Data) {
		t.Error("chunked partial-frame output differs from whole-buffer output")
	}
}

// --- empty input -------------------------------------------------------

func TestQuantizeEmpty(t *testing.T) {
	for _, bits := range []int{8, 16, 24} {
		q := testQuantizer(t, 2, QuantizeOptions{BitDepth: bits, Dither: true, Seed: 1})
		out, err := q.QuantizeFloat64(nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != 0 {
			t.Errorf("empty input produced %d samples", len(out))
		}
		out, err = q.QuantizeFloat32(make([]int, 0), nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != 0 {
			t.Errorf("empty input produced %d samples", len(out))
		}
	}
}

// --- benchmark sanity --------------------------------------------------

func BenchmarkQuantizeFloat32_16bit(b *testing.B) {
	samples := make([]float32, 44100)
	for i := range samples {
		samples[i] = float32(0.3 * math.Sin(float64(i)*0.01))
	}
	q, _ := NewQuantizer(1, QuantizeOptions{BitDepth: 16, Dither: true, ErrorFeedback: true, Seed: 1})
	dst := make([]int, len(samples))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q.QuantizeFloat32(dst, samples)
	}
}
