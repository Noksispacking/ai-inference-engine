package main

import (
	"github.com/noks/ai-inference-engine/ai/windows"
	"image"
	"image/png"
	"log"
	"os"
)

func LoadPng(path string) image.Image {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	img, err := png.Decode(file)
	if err != nil {
		log.Fatal(err)
	}
	return img
}
func main() {
	comboScene := LoadPng("C:\\Users\\32nok\\PycharmProjects\\combo_detection\\dataset\\images\\train\\coco_combo_scene_1_0.png")
	model := inference.LoadModel("C:\\Users\\32nok\\go\\src\\github.com\\noks\\ai-inference-engine\\ai\\models\\best.onnx")
	model.SetConfidence(0.5)

	boxes := model.Inference(comboScene)
	annotated := inference.AnnotateBoxes(comboScene, boxes)

	outFile, err := os.Create("annotated.png")
	if err != nil {
		panic(err)
	}
	defer outFile.Close()

	png.Encode(outFile, annotated)
}
