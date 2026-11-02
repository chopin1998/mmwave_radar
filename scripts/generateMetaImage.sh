#!/bin/bash
# Local wrapper for TI generateMetaImage.sh that uses pure-Python out2rprc.py instead of mono out2rprc.exe.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT2RPRC_PY="${SCRIPT_DIR}/out2rprc.py"

function printErrorMsg () {
  echo -e "\033[31m${1}${2}\033[0m"
}

function printInfoMsg () {
  echo -e "\033[33m${1}${2}\033[0m"
}

function out2rprc () {
    echo "python3 ${OUT2RPRC_PY} $@"
    python3 "${OUT2RPRC_PY}" "$@"
}

function makeBinFileName () {
    filename="${1%%.*}"
    ext="${1##*.}"
    echo "Out file name (without extension) is $filename and extension is $ext"
    if [ "$ext" == "bin" ]; then
        printErrorMsg "Error: $1 file passed for MSS/DSS."
        exit 1
    fi
    local __binFileNameVar=$2
    local binFileName=$filename.bin
    eval $__binFileNameVar="'$binFileName'"
}

echo "Cmd line used: $0 $@"

if [ -z "$MMWAVE_SDK_INSTALL_PATH" ]; then
    printErrorMsg "Error: MMWAVE_SDK_INSTALL_PATH needs to be defined"
    exit 1
fi
if [ $# -lt 3 ]; then
    printErrorMsg "Error: Invalid Usage. Insufficient parameters."
    exit 1
fi

FLASHIMAGE=$1
SHMEM_ALLOC=$2
MSS_IMAGE_OUT=$3
BSS_IMAGE_BIN=$4
DSS_IMAGE_OUT=$5
CONFIG_BIN=
MSS_IMAGE_BIN=
DSS_IMAGE_BIN=

MULTCOREGEN=$MMWAVE_SDK_INSTALL_PATH/scripts/ImageCreator/multicore_image_generator/MulticoreImageGen
CRC_MULTI=$MMWAVE_SDK_INSTALL_PATH/scripts/ImageCreator/crc_multicore_image/crc_multicore_image
GEN_BINCRC32=$MMWAVE_SDK_INSTALL_PATH/scripts/ImageCreator/append_bin_crc/gen_bincrc32
TEMP_FILE=$FLASHIMAGE.tmp

if [ -e "$MSS_IMAGE_OUT" ]; then
    makeBinFileName "$MSS_IMAGE_OUT" MSS_IMAGE_BIN
    out2rprc "$MSS_IMAGE_OUT" "$MSS_IMAGE_BIN"
else
    printErrorMsg "Error: Mandatory MSS image $MSS_IMAGE_OUT not found"
    exit 1
fi

if [ -e "$DSS_IMAGE_OUT" ]; then
    makeBinFileName "$DSS_IMAGE_OUT" DSS_IMAGE_BIN
    out2rprc "$DSS_IMAGE_OUT" "$DSS_IMAGE_BIN"
fi

if [ -e "$MSS_IMAGE_BIN" ]; then
    MSS_CORE="0x35510000 $MSS_IMAGE_BIN"
else
    MSS_CORE=
fi
if [ -e "$BSS_IMAGE_BIN" ]; then
    BSS_CORE="0xb5510000 $BSS_IMAGE_BIN"
else
    BSS_CORE=
fi
if [ -e "$DSS_IMAGE_BIN" ]; then
    DSS_CORE="0xd5510000 $DSS_IMAGE_BIN"
else
    DSS_CORE=
fi

echo $MULTCOREGEN LE 37 $SHMEM_ALLOC $FLASHIMAGE $CONFIG $MSS_CORE $BSS_CORE $DSS_CORE
$MULTCOREGEN LE 37 $SHMEM_ALLOC $FLASHIMAGE $CONFIG $MSS_CORE $BSS_CORE $DSS_CORE

echo $CRC_MULTI $FLASHIMAGE $TEMP_FILE
$CRC_MULTI $FLASHIMAGE $TEMP_FILE

$GEN_BINCRC32 $FLASHIMAGE

if [ -e "$FLASHIMAGE" ]; then
    printInfoMsg "**************************************************************************"
    printInfoMsg "Built the flash binary $FLASHIMAGE"
    printInfoMsg "**************************************************************************"
fi

rm -f "$TEMP_FILE" "$MSS_IMAGE_BIN" "$DSS_IMAGE_BIN" "$CONFIG_BIN"
