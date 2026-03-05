package inference

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -linference
#include "inference.h"
#include <stdlib.h>
*/
import "C"
import (
	"github.com/noks/ai-inference-engine/pkg/util"
	"image"
	"image/color"
	"image/draw"
	"unsafe"
)

type Box struct {
	X1, Y1, X2, Y2 float32
	Confidence     float32
	ClassID        int
}

type Model struct {
	ptr *C.ModelHandle
}

// LoadModel loads an ONNX model
func LoadModel(path string) *Model {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	ptr := C.LoadModel(cpath)
	if ptr == nil {
		return nil
	}
	return &Model{ptr: ptr}
}

// FreeModel releases the model
func (m *Model) Free() {
	if m.ptr != nil {
		C.FreeModel(m.ptr)
		m.ptr = nil
	}
}

// SetConfidence sets the detection threshold
func (m *Model) SetConfidence(conf float32) {
	if m.ptr != nil {
		C.SetConfidence(m.ptr, C.float(conf))
	}
}

// GetLastError returns the last error code
func (m *Model) GetLastError() int {
	if m.ptr == nil {
		return -1
	}
	return int(C.GetLastError(m.ptr))
}

// Inference runs inference on an image (raw RGB bytes)
func (m *Model) Inference(img image.Image) []Box {
	if m.ptr == nil || img == nil {
		return nil
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	imgBytes := util.RGBABytes(img)

	var count C.int
	out := C.Inference(
		m.ptr,
		(*C.uint8_t)(unsafe.Pointer(&imgBytes[0])),
		C.int(width),
		C.int(height),
		&count,
	)

	if out == nil || count == 0 {
		return nil
	}
	defer C.FreeBoxes(out)

	boxes := make([]Box, int(count))
	for i := 0; i < int(count); i++ {
		b := (*C.Box)(unsafe.Pointer(uintptr(unsafe.Pointer(out)) + uintptr(i)*unsafe.Sizeof(*out)))
		boxes[i] = Box{
			X1:         float32(b.x1),
			Y1:         float32(b.y1),
			X2:         float32(b.x2),
			Y2:         float32(b.y2),
			Confidence: float32(b.confidence),
			ClassID:    int(b.class_id),
		}
	}
	return boxes
}

func AnnotateBoxes(img image.Image, boxes []Box) *image.RGBA {
	if img == nil || len(boxes) == 0 {
		return nil
	}

	// Create RGBA copy of the image
	rgba := image.NewRGBA(img.Bounds())
	draw.Draw(rgba, rgba.Bounds(), img, image.Point{}, draw.Src)

	rectColor := color.RGBA{255, 0, 0, 255} // Red boxes
	lineWidth := 2

	for _, box := range boxes {
		x1 := int(box.X1)
		y1 := int(box.Y1)
		x2 := int(box.X2)
		y2 := int(box.Y2)

		// Draw horizontal lines
		for i := 0; i < lineWidth; i++ {
			for x := x1; x <= x2; x++ {
				if y1+i < rgba.Bounds().Dy() {
					rgba.Set(x, y1+i, rectColor)
				}
				if y2-i >= 0 {
					rgba.Set(x, y2-i, rectColor)
				}
			}
		}

		// Draw vertical lines
		for i := 0; i < lineWidth; i++ {
			for y := y1; y <= y2; y++ {
				if x1+i < rgba.Bounds().Dx() {
					rgba.Set(x1+i, y, rectColor)
				}
				if x2-i >= 0 {
					rgba.Set(x2-i, y, rectColor)
				}
			}
		}
	}

	return rgba
}
