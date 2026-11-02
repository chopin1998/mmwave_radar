#!/bin/bash
# Command-line QSPI Flash programmer for IWR6843 using TI UniFlash.
# Hardware prerequisite: Set SOP jumpers to Flashing Mode (SOP2=1, SOP1=0, SOP0=1 -> 101) and reset board.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PORT="/dev/ttyUSB0"
if [ -e "/dev/ttyACM0" ]; then
    DEFAULT_PORT="/dev/ttyACM0"
fi
PORT="${1:-${DEFAULT_PORT}}"
BIN_IMAGE="${2:-$(cd "${SCRIPT_DIR}/.." && pwd)/firmware/xwr68xx_mmw/xwr68xx_mmw_demo.bin}"

if [ -z "${UNIFLASH_DIR}" ]; then
    if [ -d "/opt/ti/uniflash_9.6.0" ]; then
        UNIFLASH_DIR="/opt/ti/uniflash_9.6.0"
    elif [ -d "/opt/ti/uniflash_9.7.0" ]; then
        UNIFLASH_DIR="/opt/ti/uniflash_9.7.0"
    else
        UNIFLASH_DIR="$(ls -d /opt/ti/uniflash_* 2>/dev/null | sort -V | tail -n 1)"
    fi
fi

CCXML_PATH="${UNIFLASH_DIR}/deskdb/content/TICloudAgent/linux/ccs_base/mmWave/gen3/IWR6843.ccxml"
if [ ! -f "${CCXML_PATH}" ]; then
    CCXML_PATH="${SCRIPT_DIR}/IWR6843.ccxml"
fi

if [ ! -f "$BIN_IMAGE" ]; then
    echo "[ERROR] Firmware binary not found: $BIN_IMAGE"
    echo "Run 'make firmware' first."
    exit 1
fi

echo "======================================================================"
echo "  IWR6843 QSPI Flash Loader (TI UniFlash CLI)"
echo "  UniFlash Dir: $UNIFLASH_DIR"
echo "  Serial Port : $PORT"
echo "  MetaImage   : $BIN_IMAGE"
echo "  Ensure SOP  : SOP[2:0] = 101 (Flashing Mode) and press NRST"
echo "======================================================================"

"${UNIFLASH_DIR}/dslite.sh" --mode flash \
    --config="${CCXML_PATH}" \
    -s "COMPort=${PORT}" \
    -e \
    -f "${BIN_IMAGE},1"
