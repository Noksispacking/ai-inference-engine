package main

import (
	"github.com/nosyliam/revolution/ai-inference-engine/platform/ai/windows"
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
	inference.LoadModel("")
}
