#! /bin/sh
if [ "$TYPE" = "GT" ]; then ./.travis-gt.sh; fi
if [ "$TYPE" = "LINT" ]; then ./.travis-lint.sh; fi
if [ "$TYPE" = "DB" ]; then ./.travis-db.sh; fi
