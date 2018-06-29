#! /bin/sh
set -e
make
if [ "$TYPE" = "GT" ] || [ "$TYPE" = "DB" ] ; then
  cd $HOME
  if [ ! -x SageMath/sage ] ; then
    rm -f SageMath.tar.bz2
    wget --progress=dot:giga $SAGE_ADDRESS$SAGE_IMAGE -O SageMath.tar.bz2
    echo "Extracting SageMath"
    tar xf SageMath.tar.bz2 --checkpoint=.1000
    #Run Sage once, so it fixes its paths as part of the installation
    SageMath/sage -v
  fi
fi
