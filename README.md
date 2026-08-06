# pygbl

A parser for the GBL and EBL firmware image formats used by the
[Silicon Labs Gecko bootloader](https://www.silabs.com/documents/public/user-guides/ug489-gecko-bootloader-user-guide-gsdk-4.pdf).

Images are parsed into structured tags, and round-trip byte-for-byte: anything the
library reads it can write back unchanged, including the trailing data that some vendors
append after the end tag.

## Installation

```console
pip install pygbl
```

The base install is pure Python and covers parsing, serializing, building images from
ELFs, and LZMA. Signing, encryption and LZ4 need optional dependencies:

```console
pip install pygbl[crypto]  # signing and encryption
pip install pygbl[lz4]     # LZ4 compression
pip install pygbl[all]
```

Calling those features without their dependency raises `MissingDependencyError`.

## Usage

Parse an image and inspect its tags:

```python
import pathlib

from pygbl import GBLApplicationInfo, parse_firmware_image

data = pathlib.Path("ncp-uart-hw.gbl").read_bytes()
image = parse_firmware_image(data)

for tag in image.tags:
    print(tag)

print(image.get_first_tag(GBLApplicationInfo).version)
print(image.get_metadata())  # opaque bytes, the schema is vendor defined

assert image.serialize() == data
```

Modify an image. Tags are frozen dataclasses, so `dataclasses.replace` works, and
`regenerate_crc` fixes up the end tag afterwards:

```python
import dataclasses

from pygbl import GBLEnd, GBLMetadata

modified = dataclasses.replace(
    image,
    tags=[t for t in image.tags if not isinstance(t, GBLEnd)]
    + [GBLMetadata(metadata=b'{"fw_type": "zigbee_ncp"}')],
).regenerate_crc()
```

Compress, encrypt and sign, in the order the bootloader expects:

```python
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from pygbl import GBLCompression

private_key = load_pem_private_key(
    pathlib.Path("vendor_sign.key").read_bytes(), password=None
)
key = bytes.fromhex("7F8FE53979B31BC556FCB131AFF42414")

sealed = image.compress(GBLCompression.LZMA).encrypt(key).sign(private_key)
pathlib.Path("signed.gbl").write_bytes(sealed.serialize(block_size=4))
```

And unwrap it again:

```python
assert sealed.verify_signature(private_key.public_key())
assert sealed.decrypt(key).decompress().serialize() == image.serialize()
```

## Building images from an ELF

Images can be built straight from a linked ELF, without `commander`. Program data comes
from the loadable segments — keyed by *physical* address, since initialized data is
stored in flash but linked at its RAM address — and the application info tag is read
from the SDK's `application_properties_t` struct:

```python
from pygbl import build_application_gbl, build_bootloader_gbl

with open("zigbee_ncp.out", "rb") as f:
    image = build_application_gbl(f, metadata=b'{"fw_type": "zigbee_ncp"}')

with open("bootloader.out", "rb") as f:
    bootloader = build_bootloader_gbl(f)
```

Both reproduce `commander gbl create` byte-for-byte. Bootloader payloads carry a CRC32
of themselves, which `build_bootloader_gbl` appends for you.

## EBL

EBL images, used by older EM3xx parts, work the same way:

```python
from pygbl import EBLEraseProgram, parse_firmware_image

image = parse_firmware_image(pathlib.Path("ncp-uart-sw.ebl").read_bytes())

for tag in image.get_tags(EBLEraseProgram):
    print(f"{tag.address:#010x} {len(tag.data)} bytes")
```

## Bootloader and application images

A GBL can contain a bootloader, an application, or both. Combined images can be split
apart and recombined, which is useful because some bootloaders cannot flash a combined
image in one pass:

```python
bootloader, application = combined.split_bootloader_app()
recombined = application.combine_bootloader_app(bootloader)
```
