# Top-level Makefile for IWR6843 mmWave Radar Project
FIRMWARE_DIR := firmware/xwr68xx_mmw
UV           ?= $(shell command -v uv 2>/dev/null || ([ -x /home/marco/.local/bin/uv ] && echo /home/marco/.local/bin/uv) || echo /home/marco/.cargo/bin/uv)
PORT         ?= $(shell [ -e /dev/ttyACM0 ] && echo /dev/ttyACM0 || echo /dev/ttyUSB0)

.PHONY: all firmware clean mss dss flash gui gui-hw pty-emulator test-host help

all: firmware

firmware:
	$(MAKE) -C $(FIRMWARE_DIR) all

mss:
	$(MAKE) -C $(FIRMWARE_DIR) mssDemo

dss:
	$(MAKE) -C $(FIRMWARE_DIR) dssDemo

clean:
	$(MAKE) -C $(FIRMWARE_DIR) clean

flash:
	./scripts/flash_uniflash.sh $(PORT) $(FIRMWARE_DIR)/xwr68xx_mmw_demo.bin

gui:
	$(UV) run python -m host.gui_visualizer --mock --auto-start

gui-hw:
	$(UV) run python -m host.gui_visualizer --hw

pty-emulator:
	$(UV) run python -m host.mock_radar_pty

test-host:
	$(UV) run python -m host.gui_visualizer --headless-test 50

help:
	@echo "Targets:"
	@echo "  make firmware             - Build MSS + DSS + MetaImage (.bin)"
	@echo "  make flash PORT=/dev/...  - Flash xwr68xx_mmw_demo.bin via UniFlash CLI"
	@echo "  make gui                  - Launch PySide6/PyQtGraph GUI in Mock Simulation mode"
	@echo "  make gui-hw               - Launch GUI in Hardware / PTY Dual-UART mode"
	@echo "  make pty-emulator         - Start Linux PTY Dual-UART Radar Emulator (/tmp/ttyMockRadar*)"
	@echo "  make test-host            - Run 50-frame fragmented TLV stream self-test"
