#! /bin/sh
set -e
make
if [ "$TYPE" = "GT" ]; then
  cd $HOME
  if [ ! -x SageMath/sage ] ; then
    rm -f SageMath.tar.bz2
    wget --progress=dot:giga $SAGE_ADDRESS$SAGE_IMAGE -O SageMath.tar.bz2
    echo "Extracting SageMath"
    tar xf SageMath.tar.bz2 --checpoint=.1000
  fi
fi
