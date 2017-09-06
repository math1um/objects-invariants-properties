.PHONY: build dist clean clear
build:
	mkdir -p build
	cp -r src/* build

dist: build
	mkdir -p dist
	cd build && zip -r ../dist/objects-invariants-properties.zip *

clean:
	rm -r build

clear: clean
	rm -r dist

