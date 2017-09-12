SOURCEFILES = src/gt_precomputed_database.sage
GT_COMPONENTS = src/Utilities/utilities.sage src/Utilities/final.sage \
src/Invariants/invariants.sage  src/Invariants/final.sage \
src/Properties/properties.sage src/Properties/final.sage \
src/Theorems/theorems.sage src/Theorems/final.sage \
src/Objects/graphs.sage src/Objects/class0graphs.sage src/Objects/sloanegraphs.sage \
src/Objects/dimacsgraphs.sage src/Objects/final.sage \
src/final.sage


.PHONY: build db dist clean clear
build: db
	mkdir -p build
	cp -r $(SOURCEFILES) build
	cat $(GT_COMPONENTS) > build/gt.sage

db:
	mkdir -p build
	rm -f build/gt_precomputed_database.db
	sqlite3 build/gt_precomputed_database.db < db/tables.sql
	cat db/begin_transaction.sql db/invariants/*.sql db/commit.sql | sqlite3 build/gt_precomputed_database.db
	cat db/begin_transaction.sql db/properties/*.sql db/commit.sql | sqlite3 build/gt_precomputed_database.db

dist: build
	mkdir -p dist
	cd build && zip -r ../dist/objects-invariants-properties.zip *

clean:
	rm -rf build

clear: clean
	rm -rf dist
