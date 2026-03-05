package inference

/*
#cgo CFLAGS: -I.
#cgo LDFLAGS: -L. -linference
#include "inference.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

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
func (m *Model) Inference(img []byte, width, height int) []Box {
	if m.ptr == nil || len(img) == 0 {
		return nil
	}

	var count C.int
	out := C.Inference(
		m.ptr,
		(*C.uint8_t)(unsafe.Pointer(&img[0])),
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
