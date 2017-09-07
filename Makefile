.PHONY: build db dist clean clear
build: db
	mkdir -p build
	cp -r src/* build

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
	rm -r build

clear: clean
	rm -r dist
