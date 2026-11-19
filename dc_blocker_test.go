package audio

import (
	"math"
	"testing"
)

const (
	dcTestSampleRate = 48000
	dcTestFrequency  = 300.0
	dcTestCutoff     = 20.0
	// 4s of audio: at a 20Hz cutoff the filter time constant is ~2100
	// samples, so 19200 frames leave ~9 time constants for the DC transient
	// to decay.
	dcTestFrames = 19200
)

// makeStereoSineWithBias builds an interleaved stereo float64 signal made of
// a low frequency sine plus a constant, channel specific, DC bias.
func makeStereoSineWithBias(frames int, sr int, freq float64, biasL, biasR, amp float64) []float64 {
	data := make([]float64, frames*2)
	for i := 0; i < frames; i++ {
		s := amp * math.Sin(2*math.Pi*freq*float64(i)/float64(sr))
		data[i*2] = s + biasL
		data[i*2+1] = s + biasR
	}
	return data
}

// mean returns the arithmetic mean of the samples belonging to one channel.
func channelMean(data []float64, nCh, ch int) float64 {
	sum := 0.0
	n := 0
	for i := ch; i < len(data); i += nCh {
		sum += data[i]
		n++
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// rmsSineAmplitude estimates the sine amplitude of a channel once its mean
// has been removed.
func channelRMS(data []float64, nCh, ch int) float64 {
	mean := channelMean(data, nCh, ch)
	sum := 0.0
	n := 0
	for i := ch; i < len(data); i += nCh {
		d := data[i] - mean
		sum += d * d
		n++
	}
	if n == 0 {
		return 0
	}
	return math.Sqrt(sum / float64(n) * 2)
}

func TestNewDCBlocker_Errors(t *testing.T) {
	tests := []struct {
		name    string
		cutoff  float64
		format  *Format
		wantErr error
	}{
		{"nil format", 20, nil, ErrInvalidBuffer},
		{"no channel", 20, &Format{NumChannels: 0, SampleRate: 48000}, ErrInvalidBuffer},
		{"negative channels", 20, &Format{NumChannels: -1, SampleRate: 48000}, ErrInvalidBuffer},
		{"zero sample rate", 20, &Format{NumChannels: 1, SampleRate: 0}, ErrInvalidSampleRate},
		{"negative sample rate", 20, &Format{NumChannels: 1, SampleRate: -48000}, ErrInvalidSampleRate},
		{"zero cutoff", 0, FormatMono48000, ErrInvalidCutoff},
		{"negative cutoff", -20, FormatMono48000, ErrInvalidCutoff},
		{"NaN cutoff", math.NaN(), FormatMono48000, ErrInvalidCutoff},
		{"+Inf cutoff", math.Inf(1), FormatMono48000, ErrInvalidCutoff},
		{"-Inf cutoff", math.Inf(-1), FormatMono48000, ErrInvalidCutoff},
		{"cutoff at nyquist", 24000, FormatMono48000, ErrInvalidCutoff},
		{"cutoff above nyquist", 30000, FormatMono48000, ErrInvalidCutoff},
		{"valid low cutoff", 1, FormatMono48000, nil},
		{"valid cutoff just under nyquist", 23999.999, FormatMono48000, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, err := NewDCBlocker(tt.cutoff, tt.format)
			if tt.wantErr != nil {
				if err != tt.wantErr {
					t.Fatalf("NewDCBlocker() error = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("NewDCBlocker() unexpected error: %v", err)
			}
			if d == nil {
				t.Fatal("NewDCBlocker() returned a nil blocker")
			}
			if d.a1 <= -1 || d.a1 >= 1 {
				t.Fatalf("pole coefficient %v is not strictly in (-1, 1), the filter would be unstable", d.a1)
			}
		})
	}
}

func TestDCBlocker_FloatBuffer(t *testing.T) {
	const (
		biasL = 0.35
		biasR = -0.22
		amp   = 0.25
	)
	src := &FloatBuffer{
		Format: FormatStereo48000,
		Data:   makeStereoSineWithBias(dcTestFrames, dcTestSampleRate, dcTestFrequency, biasL, biasR, amp),
	}
	original := make([]float64, len(src.Data))
	copy(original, src.Data)

	blocker, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}

	// Whole buffer at once.
	wholeBuf, err := blocker.ProcessFloatBuffer(src)
	if err != nil {
		t.Fatalf("ProcessFloatBuffer() error: %v", err)
	}
	if wholeBuf == src {
		t.Fatal("the input buffer must not be returned in place")
	}
	if wholeBuf.Format.NumChannels != src.Format.NumChannels || wholeBuf.Format.SampleRate != src.Format.SampleRate {
		t.Fatalf("output format = %+v, want %+v", wholeBuf.Format, src.Format)
	}
	if len(wholeBuf.Data) != len(src.Data) {
		t.Fatalf("output length = %d, want %d", len(wholeBuf.Data), len(src.Data))
	}
	// The source buffer must remain untouched.
	for i := range original {
		if src.Data[i] != original[i] {
			t.Fatalf("source buffer was modified at index %d: %v != %v", i, src.Data[i], original[i])
		}
	}

	// The steady state mean of both channels must be ~0, their independent
	// biases having been removed. Skip the transient start of the signal:
	// at a 20Hz cutoff the filter time constant is ~2100 samples.
	start := len(wholeBuf.Data) / 2
	for ch := 0; ch < 2; ch++ {
		mean := channelMean(wholeBuf.Data[start:], 2, ch)
		if math.Abs(mean) > 1e-3 {
			t.Errorf("channel %d steady state mean = %v, want ~0", ch, mean)
		}
		// The low frequency tone must survive the filter almost unchanged.
		gotAmp := channelRMS(wholeBuf.Data[start:], 2, ch)
		if math.Abs(gotAmp-amp) > amp*0.02 {
			t.Errorf("channel %d tone amplitude = %v, want ~%v", ch, gotAmp, amp)
		}
	}

	// Any chunking of the input must produce exactly the same per-sample
	// output as the single call above.
	chunkSizes := []int{1, 2, 3, 5, 7, 16, 31, 64, 127, 256, 1000, 4096}
	for _, chunkFrames := range chunkSizes {
		chunkSamples := chunkFrames * 2
		b, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
		if err != nil {
			t.Fatalf("NewDCBlocker() error: %v", err)
		}
		var got []float64
		for off := 0; off < len(src.Data); off += chunkSamples {
			end := off + chunkSamples
			if end > len(src.Data) {
				end = len(src.Data)
			}
			out, err := b.ProcessFloatBuffer(&FloatBuffer{Format: FormatStereo48000, Data: src.Data[off:end]})
			if err != nil {
				t.Fatalf("chunk size %d: ProcessFloatBuffer() error: %v", chunkFrames, err)
			}
			got = append(got, out.Data...)
		}
		if len(got) != len(wholeBuf.Data) {
			t.Fatalf("chunk size %d: got %d samples, want %d", chunkFrames, len(got), len(wholeBuf.Data))
		}
		for i := range got {
			if got[i] != wholeBuf.Data[i] {
				t.Fatalf("chunk size %d: sample %d differs: %v != %v", chunkFrames, i, got[i], wholeBuf.Data[i])
			}
		}
	}
}

func TestDCBlocker_Float32Buffer(t *testing.T) {
	src64 := makeStereoSineWithBias(dcTestFrames, dcTestSampleRate, dcTestFrequency, 0.4, -0.3, 0.2)
	src := &Float32Buffer{
		Format:         FormatStereo48000,
		Data:           make([]float32, len(src64)),
		SourceBitDepth: 24,
	}
	for i, v := range src64 {
		src.Data[i] = float32(v)
	}

	blocker, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	whole, err := blocker.ProcessFloat32Buffer(src)
	if err != nil {
		t.Fatalf("ProcessFloat32Buffer() error: %v", err)
	}
	if whole.SourceBitDepth != src.SourceBitDepth {
		t.Errorf("SourceBitDepth = %d, want %d", whole.SourceBitDepth, src.SourceBitDepth)
	}
	if len(whole.Data) != len(src.Data) {
		t.Fatalf("output length = %d, want %d", len(whole.Data), len(src.Data))
	}

	// Cross chunk consistency in float32.
	b, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	var got []float32
	for off := 0; off < len(src.Data); off += 6 {
		end := off + 6
		if end > len(src.Data) {
			end = len(src.Data)
		}
		out, err := b.ProcessFloat32Buffer(&Float32Buffer{Format: FormatStereo48000, Data: src.Data[off:end], SourceBitDepth: src.SourceBitDepth})
		if err != nil {
			t.Fatalf("ProcessFloat32Buffer() error: %v", err)
		}
		got = append(got, out.Data...)
	}
	if len(got) != len(whole.Data) {
		t.Fatalf("got %d samples, want %d", len(got), len(whole.Data))
	}
	for i := range got {
		if got[i] != whole.Data[i] {
			t.Fatalf("sample %d differs: %v != %v", i, got[i], whole.Data[i])
		}
	}

	// Non finite float32 samples.
	nanBuf := &Float32Buffer{Format: FormatStereo48000, Data: []float32{1, 1, float32(math.NaN()), 1}}
	if _, err := blocker.ProcessFloat32Buffer(nanBuf); err != ErrNonFiniteSample {
		t.Fatalf("NaN input error = %v, want %v", err, ErrNonFiniteSample)
	}
	infBuf := &Float32Buffer{Format: FormatStereo48000, Data: []float32{1, 1, float32(math.Inf(1)), 1}}
	if _, err := blocker.ProcessFloat32Buffer(infBuf); err != ErrNonFiniteSample {
		t.Fatalf("Inf input error = %v, want %v", err, ErrNonFiniteSample)
	}
}

func TestDCBlocker_IntBuffer(t *testing.T) {
	const (
		biasL = 4000
		biasR = -2500
		amp   = 3000
	)
	src := &IntBuffer{
		Format:         FormatStereo48000,
		SourceBitDepth: 16,
		Data:           make([]int, dcTestFrames*2),
	}
	for i := 0; i < dcTestFrames; i++ {
		s := int(amp * math.Sin(2*math.Pi*dcTestFrequency*float64(i)/float64(dcTestSampleRate)))
		src.Data[i*2] = s + biasL
		src.Data[i*2+1] = s + biasR
	}
	original := make([]int, len(src.Data))
	copy(original, src.Data)

	blocker, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	whole, err := blocker.ProcessIntBuffer(src)
	if err != nil {
		t.Fatalf("ProcessIntBuffer() error: %v", err)
	}
	if whole.SourceBitDepth != src.SourceBitDepth {
		t.Errorf("SourceBitDepth = %d, want %d", whole.SourceBitDepth, src.SourceBitDepth)
	}
	for i := range original {
		if src.Data[i] != original[i] {
			t.Fatalf("source buffer was modified at index %d", i)
		}
	}
	// Steady state mean close to 0 (integer rounding allows a small slack).
	startFrames := dcTestFrames * 3 / 4
	for ch := 0; ch < 2; ch++ {
		sum, n := 0, 0
		for f := startFrames; f < dcTestFrames; f++ {
			sum += whole.Data[f*2+ch]
			n++
		}
		mean := float64(sum) / float64(n)
		if math.Abs(mean) > 1.5 {
			t.Errorf("channel %d steady state mean = %v, want ~0", ch, mean)
		}
	}

	// Chunked processing must match the whole processing sample for sample.
	b, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	var got []int
	for off := 0; off < len(src.Data); off += 10 {
		end := off + 10
		if end > len(src.Data) {
			end = len(src.Data)
		}
		out, err := b.ProcessIntBuffer(&IntBuffer{Format: FormatStereo48000, Data: src.Data[off:end], SourceBitDepth: 16})
		if err != nil {
			t.Fatalf("ProcessIntBuffer() error: %v", err)
		}
		got = append(got, out.Data...)
	}
	if len(got) != len(whole.Data) {
		t.Fatalf("got %d samples, want %d", len(got), len(whole.Data))
	}
	for i := range got {
		if got[i] != whole.Data[i] {
			t.Fatalf("sample %d differs: %d != %d", i, got[i], whole.Data[i])
		}
	}
}

func TestSaturatedInt(t *testing.T) {
	tests := []struct {
		name     string
		v        float64
		bitDepth int
		want     int
	}{
		{"16bit in range", 123.6, 16, 124},
		{"16bit negative half rounds away from zero", -123.5, 16, -124},
		{"16bit positive half rounds away from zero", 123.5, 16, 124},
		{"16bit positive saturation", 40000, 16, 32767},
		{"16bit negative saturation", -40000, 16, -32768},
		{"8bit positive saturation", 200, 8, 127},
		{"8bit negative saturation", -200, 8, -128},
		{"24bit positive saturation", 1e7, 24, 8388607},
		{"24bit negative saturation", -1e7, 24, -8388608},
		{"32bit positive saturation", 3e9, 32, 2147483647},
		{"32bit negative saturation", -3e9, 32, -2147483648},
		{"64bit positive saturation", 1e20, 64, math.MaxInt64},
		{"64bit negative saturation", -1e20, 64, math.MinInt64},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := saturatedInt(tt.v, tt.bitDepth); got != tt.want {
				t.Errorf("saturatedInt(%v, %d) = %d, want %d", tt.v, tt.bitDepth, got, tt.want)
			}
		})
	}

	// Filtering an int16 buffer whose values sit at the format extremes must
	// never produce values outside of the int16 range (no wrap around).
	blocker, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	extremes := &IntBuffer{
		Format:         FormatStereo48000,
		SourceBitDepth: 16,
		Data: []int{
			32767, -32768, 32767, -32768,
			-32768, 32767, -32768, 32767,
		},
	}
	out, err := blocker.ProcessIntBuffer(extremes)
	if err != nil {
		t.Fatalf("ProcessIntBuffer() error: %v", err)
	}
	for i, v := range out.Data {
		if v < -32768 || v > 32767 {
			t.Errorf("sample %d = %d is outside of the int16 range", i, v)
		}
	}
}

func TestDCBlocker_Reset(t *testing.T) {
	src := &FloatBuffer{
		Format: FormatMono48000,
		Data:   make([]float64, 64),
	}
	for i := range src.Data {
		src.Data[i] = 0.5
	}

	blocker, err := NewDCBlocker(dcTestCutoff, FormatMono48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	first, err := blocker.ProcessFloatBuffer(src)
	if err != nil {
		t.Fatalf("ProcessFloatBuffer() error: %v", err)
	}
	if blocker.xPrev[0] == 0 && blocker.yPrev[0] == 0 {
		t.Fatal("the filter state should have moved away from zero")
	}

	blocker.Reset()
	if blocker.xPrev[0] != 0 || blocker.yPrev[0] != 0 {
		t.Fatalf("Reset() did not clear the state: x=%v y=%v", blocker.xPrev[0], blocker.yPrev[0])
	}
	if len(blocker.carry) != 0 {
		t.Fatal("Reset() did not clear the buffered incomplete frame")
	}
	second, err := blocker.ProcessFloatBuffer(src)
	if err != nil {
		t.Fatalf("ProcessFloatBuffer() error: %v", err)
	}
	for i := range first.Data {
		if first.Data[i] != second.Data[i] {
			t.Fatalf("after Reset() sample %d = %v, want %v", i, second.Data[i], first.Data[i])
		}
	}
}

func TestDCBlocker_ChannelsAreIndependent(t *testing.T) {
	// Two channels with opposite, large biases: their states must not bleed
	// into each other.
	data := makeStereoSineWithBias(2400, dcTestSampleRate, dcTestFrequency, 0.8, -0.8, 0.1)
	src := &FloatBuffer{Format: FormatStereo48000, Data: data}

	blocker, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	out, err := blocker.ProcessFloatBuffer(src)
	if err != nil {
		t.Fatalf("ProcessFloatBuffer() error: %v", err)
	}
	// A mono blocker fed the same single channel signal must match each
	// stereo channel individually.
	for ch := 0; ch < 2; ch++ {
		monoData := make([]float64, len(data)/2)
		for i := range monoData {
			monoData[i] = data[i*2+ch]
		}
		monoBlocker, err := NewDCBlocker(dcTestCutoff, FormatMono48000)
		if err != nil {
			t.Fatalf("NewDCBlocker() error: %v", err)
		}
		monoOut, err := monoBlocker.ProcessFloatBuffer(&FloatBuffer{Format: FormatMono48000, Data: monoData})
		if err != nil {
			t.Fatalf("ProcessFloatBuffer() error: %v", err)
		}
		for i := range monoOut.Data {
			if monoOut.Data[i] != out.Data[i*2+ch] {
				t.Fatalf("channel %d sample %d: stereo result %v differs from mono result %v", ch, i, out.Data[i*2+ch], monoOut.Data[i])
			}
		}
	}
}

func TestDCBlocker_IncompleteFrames(t *testing.T) {
	// 10 full stereo frames plus one trailing sample (21 samples), fed in
	// chunks of 3 samples: every chunk boundary cuts through a frame, so the
	// trailing samples must be carried over. The output for the 10 complete
	// frames must match a single, frame-aligned call exactly.
	full := makeStereoSineWithBias(10, dcTestSampleRate, dcTestFrequency, 0.3, -0.15, 0.1)
	src := &FloatBuffer{
		Format: FormatStereo48000,
		Data:   append(full, 0.42),
	}
	wholeBlocker, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	whole, err := wholeBlocker.ProcessFloatBuffer(&FloatBuffer{Format: FormatStereo48000, Data: full})
	if err != nil {
		t.Fatalf("ProcessFloatBuffer() error: %v", err)
	}

	streaming, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	var got []float64
	for off := 0; off < len(src.Data); off += 3 {
		end := off + 3
		if end > len(src.Data) {
			end = len(src.Data)
		}
		out, err := streaming.ProcessFloatBuffer(&FloatBuffer{Format: FormatStereo48000, Data: src.Data[off:end]})
		if err != nil {
			t.Fatalf("ProcessFloatBuffer() error: %v", err)
		}
		got = append(got, out.Data...)
	}
	// The trailing sample belongs to an incomplete 11th frame and is
	// retained, so only the 20 complete samples come back.
	if len(got) != len(whole.Data) {
		t.Fatalf("got %d complete samples, want %d", len(got), len(whole.Data))
	}
	for i := range got {
		if got[i] != whole.Data[i] {
			t.Fatalf("sample %d differs: %v != %v", i, got[i], whole.Data[i])
		}
	}
	if len(streaming.carry) != 1 {
		t.Fatalf("got %d buffered samples, want 1", len(streaming.carry))
	}
	// Completing the frame releases the carried sample and keeps continuity.
	out, err := streaming.ProcessFloatBuffer(&FloatBuffer{Format: FormatStereo48000, Data: []float64{-0.07}})
	if err != nil {
		t.Fatalf("ProcessFloatBuffer() error: %v", err)
	}
	if len(out.Data) != 2 || len(streaming.carry) != 0 {
		t.Fatalf("completing the frame should release 2 samples and clear the carry, got %d out / %d carry", len(out.Data), len(streaming.carry))
	}
}

func TestDCBlocker_EmptyAndInvalidBuffers(t *testing.T) {
	blocker, err := NewDCBlocker(dcTestCutoff, FormatStereo48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}

	// Empty buffers produce empty output and must not touch the filter state.
	for _, buf := range []Buffer{
		&FloatBuffer{Format: FormatStereo48000, Data: []float64{}},
		&Float32Buffer{Format: FormatStereo48000, Data: []float32{}},
		&IntBuffer{Format: FormatStereo48000, Data: []int{}},
	} {
		out, err := blocker.Process(buf)
		if err != nil {
			t.Fatalf("%T: Process() error: %v", buf, err)
		}
		if out.NumFrames() != 0 {
			t.Errorf("%T: got %d frames, want 0", buf, out.NumFrames())
		}
	}

	// Nil and malformed buffers.
	if _, err := blocker.Process(nil); err != ErrInvalidBuffer {
		t.Errorf("nil buffer error = %v, want %v", err, ErrInvalidBuffer)
	}
	if _, err := blocker.Process(&FloatBuffer{Data: []float64{1, 2}}); err != ErrInvalidBuffer {
		t.Errorf("nil format error = %v, want %v", err, ErrInvalidBuffer)
	}
	if _, err := blocker.ProcessFloatBuffer(nil); err != ErrInvalidBuffer {
		t.Errorf("nil FloatBuffer error = %v, want %v", err, ErrInvalidBuffer)
	}

	// Format mismatches.
	if _, err := blocker.ProcessFloatBuffer(&FloatBuffer{Format: FormatMono48000, Data: []float64{1}}); err != ErrChannelMismatch {
		t.Errorf("channel mismatch error = %v, want %v", err, ErrChannelMismatch)
	}
	if _, err := blocker.ProcessFloatBuffer(&FloatBuffer{Format: FormatStereo44100, Data: []float64{1, 2}}); err != ErrSampleRateMismatch {
		t.Errorf("sample rate mismatch error = %v, want %v", err, ErrSampleRateMismatch)
	}

	// Non finite float64 samples.
	nanBuf := &FloatBuffer{Format: FormatStereo48000, Data: []float64{1, 1, math.NaN(), 1}}
	if _, err := blocker.ProcessFloatBuffer(nanBuf); err != ErrNonFiniteSample {
		t.Errorf("NaN error = %v, want %v", err, ErrNonFiniteSample)
	}
	infBuf := &FloatBuffer{Format: FormatStereo48000, Data: []float64{1, 1, math.Inf(-1), 1}}
	if _, err := blocker.ProcessFloatBuffer(infBuf); err != ErrNonFiniteSample {
		t.Errorf("Inf error = %v, want %v", err, ErrNonFiniteSample)
	}
	// A rejected buffer must not have advanced the state.
	if blocker.xPrev[0] != 0 || blocker.yPrev[0] != 0 {
		t.Fatal("the state changed while processing an invalid buffer")
	}
}

func TestDCBlocker_GenericProcessAndPCMBuffer(t *testing.T) {
	blocker, err := NewDCBlocker(dcTestCutoff, FormatMono48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	src := &FloatBuffer{Format: FormatMono48000, Data: make([]float64, 128)}
	for i := range src.Data {
		src.Data[i] = 0.7
	}
	out, err := blocker.Process(src)
	if err != nil {
		t.Fatalf("Process() error: %v", err)
	}
	if _, ok := out.(*FloatBuffer); !ok {
		t.Fatalf("Process() returned %T, want *FloatBuffer", out)
	}

	// A PCMBuffer goes through its float64 representation.
	pcm := &PCMBuffer{
		Format:   FormatMono48000,
		DataType: DataTypeF64,
		F64:      append([]float64{}, src.Data...),
	}
	pcmBlocker, err := NewDCBlocker(dcTestCutoff, FormatMono48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	pcmOut, err := pcmBlocker.Process(pcm)
	if err != nil {
		t.Fatalf("Process(PCMBuffer) error: %v", err)
	}
	fb, ok := pcmOut.(*FloatBuffer)
	if !ok {
		t.Fatalf("Process(PCMBuffer) returned %T, want *FloatBuffer", pcmOut)
	}
	for i := range fb.Data {
		if fb.Data[i] != out.(*FloatBuffer).Data[i] {
			t.Fatalf("PCMBuffer path sample %d = %v, want %v", i, fb.Data[i], out.(*FloatBuffer).Data[i])
		}
	}
}

func TestDCBlocker_DCOnlySignal(t *testing.T) {
	// A pure constant must converge to (almost) zero.
	blocker, err := NewDCBlocker(dcTestCutoff, FormatMono48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	data := make([]float64, 4800)
	for i := range data {
		data[i] = -1.5
	}
	out, err := blocker.ProcessFloatBuffer(&FloatBuffer{Format: FormatMono48000, Data: data})
	if err != nil {
		t.Fatalf("ProcessFloatBuffer() error: %v", err)
	}
	if math.Abs(out.Data[len(out.Data)-1]) > 1e-5 {
		t.Fatalf("last sample of a DC only signal = %v, want ~0", out.Data[len(out.Data)-1])
	}
}

func TestDCBlocker_IntBitDepthInference(t *testing.T) {
	tests := []struct {
		name string
		data []int
		want int
	}{
		{"empty defaults to 8", []int{}, 8},
		{"8bit", []int{10, -12, 127}, 8},
		{"16bit", []int{1000, -2000}, 16},
		{"24bit", []int{40000}, 24},
		{"32bit", []int{9000000}, 32},
		{"64bit", []int{int(4294967296) * 2}, 64},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := &IntBuffer{Format: FormatMono48000, Data: tt.data}
			if got := intBufferBitDepth(buf); got != tt.want {
				t.Errorf("intBufferBitDepth() = %d, want %d", got, tt.want)
			}
			// The buffer must not have been mutated by the inference.
			if buf.SourceBitDepth != 0 {
				t.Errorf("SourceBitDepth was mutated to %d", buf.SourceBitDepth)
			}
			// An explicitly set bit depth is always honored.
			buf.SourceBitDepth = 24
			if got := intBufferBitDepth(buf); got != 24 {
				t.Errorf("intBufferBitDepth() with explicit depth = %d, want 24", got)
			}
		})
	}
}

func TestDCBlocker_NilReceiversAndTypedNil(t *testing.T) {
	var d *DCBlocker
	d.Reset() // must not panic on a nil receiver

	blocker, err := NewDCBlocker(dcTestCutoff, FormatMono48000)
	if err != nil {
		t.Fatalf("NewDCBlocker() error: %v", err)
	}
	var nilFloat *FloatBuffer
	if _, err := blocker.ProcessFloatBuffer(nilFloat); err != ErrInvalidBuffer {
		t.Errorf("typed nil FloatBuffer error = %v, want %v", err, ErrInvalidBuffer)
	}
	var nilF32 *Float32Buffer
	if _, err := blocker.ProcessFloat32Buffer(nilF32); err != ErrInvalidBuffer {
		t.Errorf("typed nil Float32Buffer error = %v, want %v", err, ErrInvalidBuffer)
	}
	var nilInt *IntBuffer
	if _, err := blocker.ProcessIntBuffer(nilInt); err != ErrInvalidBuffer {
		t.Errorf("typed nil IntBuffer error = %v, want %v", err, ErrInvalidBuffer)
	}
	var nilPCM *PCMBuffer
	if _, err := blocker.Process(nilPCM); err != ErrInvalidBuffer {
		t.Errorf("typed nil PCMBuffer error = %v, want %v", err, ErrInvalidBuffer)
	}
	// Unknown Buffer implementation.
	if _, err := blocker.Process(unknownBuffer{}); err != ErrInvalidBuffer {
		t.Errorf("unknown buffer error = %v, want %v", err, ErrInvalidBuffer)
	}
}

// unknownBuffer is a Buffer implementation not handled by DCBlocker.Process.
type unknownBuffer struct{}

func (unknownBuffer) PCMFormat() *Format              { return FormatMono48000 }
func (unknownBuffer) NumFrames() int                  { return 0 }
func (unknownBuffer) AsFloatBuffer() *FloatBuffer     { return nil }
func (unknownBuffer) AsFloat32Buffer() *Float32Buffer { return nil }
func (unknownBuffer) AsIntBuffer() *IntBuffer         { return nil }
func (unknownBuffer) Clone() Buffer                   { return nil }
