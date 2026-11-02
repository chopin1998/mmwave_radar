#!/usr/bin/env python3
"""
Pure-Python replacement for TI's out2rprc.exe (converts 32-bit ELF .xer4f/.xe674 to TI RPRC .bin).
Eliminates the Mono (.NET) dependency on Linux while producing byte-for-byte identical RPRC output.
"""
import struct
import sys


def elf_to_rprc(elf_path: str, rprc_path: str) -> None:
    with open(elf_path, "rb") as f:
        data = f.read()

    if data[:4] != b"\x7fELF":
        raise ValueError(f"{elf_path} is not a valid ELF file")

    (
        e_entry,
        e_phoff,
        e_shoff,
        e_flags,
        e_ehsize,
        e_phentsize,
        e_phnum,
        e_shentsize,
        e_shnum,
        e_shstrndx,
    ) = struct.unpack("<IIIIHHHHHH", data[24:52])

    SHT_PROGBITS = 1
    SHF_ALLOC = 0x2

    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        (
            sh_name,
            sh_type,
            sh_flags,
            sh_addr,
            sh_offset,
            sh_size,
            sh_link,
            sh_info,
            sh_addralign,
            sh_entsize,
        ) = struct.unpack("<IIIIIIIIII", data[off : off + 40])
        if sh_type == SHT_PROGBITS and (sh_flags & SHF_ALLOC) and sh_size > 0:
            sections.append((sh_addr, sh_offset, sh_size))

    sections.sort(key=lambda s: s[0])

    out = bytearray()
    # 24-byte RPRC file header: magic ("RPRC"), entry, rsvd0(0), num_sections, version(1), rsvd1(0)
    out += struct.pack("<4sIIIII", b"RPRC", e_entry, 0, len(sections), 1, 0)

    for sh_addr, sh_offset, sh_size in sections:
        size4 = (sh_size + 3) & ~3
        size8 = (size4 + 7) & ~7
        sec_data = data[sh_offset : sh_offset + sh_size] + b"\x00" * (size8 - sh_size)
        # 24-byte RPRC section header: run_addr, rsvd0(0), size(4-byte aligned), rsvd_crc(0), rsvd1(0), rsvd2(0)
        out += struct.pack("<IIIIII", sh_addr, 0, size4, 0, 0, 0)
        out += sec_data

    with open(rprc_path, "wb") as f:
        f.write(out)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.elf> <output.rprc.bin>", file=sys.stderr)
        sys.exit(1)
    elf_to_rprc(sys.argv[1], sys.argv[2])
