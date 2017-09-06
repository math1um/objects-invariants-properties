.PHONY: build clean
build:
	mkdir -p build
	cp -r src/* build

clean:
	rm -r build
