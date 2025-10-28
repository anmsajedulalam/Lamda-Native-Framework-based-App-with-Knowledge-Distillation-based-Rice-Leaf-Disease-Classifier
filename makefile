# Makefile for Lambda Native Mobile App
APPNAME = mobilenetclassifier
APPSOURCES = main.scm ui/layout.scm
MODELS = model_quantized.tflite class_names.json

# Platform targets
android: $(APPSOURCES) $(MODELS)
	ln -sf $(MODELS) plugins/
	build-mobilenet -t android $(APPNAME)

ios: $(APPSOURCES) $(MODELS)
	ln -sf $(MODELS) plugins/
	build-mobilenet -t ios $(APPNAME)

# Clean build
clean:
	rm -rf build plugins/$(MODELS)

# Install dependencies
setup:
	conda env create -f environment.yml