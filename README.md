# pygbl

A parser for the EBL and GBL (v3 and v4) firmware image formats used by the
[Silicon Labs Gecko bootloader](https://www.silabs.com/documents/public/user-guides/ug489-gecko-bootloader-user-guide-gsdk-4.pdf).

Images are parsed into structured tags and round-trip byte-for-byte, with anything
following the end tag handed back separately rather than folded into the image.

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

from pygbl import GBL3ApplicationInfo, parse_firmware_image

data = pathlib.Path("ncp-uart-hw.gbl").read_bytes()
image = parse_firmware_image(data)

for tag in image.tags:
    print(tag)

print(image.get_first_tag(GBL3ApplicationInfo).version)
print(image.get_metadata())  # opaque bytes, the schema is vendor defined
```

`serialize` pads up to a word boundary, as `commander` does. `parse_firmware_image`
discards anything after the end tag; `deserialize` hands it back:

```python
image, trailing = GBL3Image.deserialize(data)

assert image.serialize(block_size=1) + trailing == data
```

Modify an image. Tags are frozen dataclasses, so `dataclasses.replace` works, and
`regenerate_crc` fixes up the end tag afterwards:

```python
import dataclasses

from pygbl import GBL3End, GBL3Metadata

modified = dataclasses.replace(
    image,
    tags=[t for t in image.tags if not isinstance(t, GBL3End)]
    + [GBL3Metadata(metadata=b'{"fw_type": "zigbee_ncp"}')],
).regenerate_crc()
```

Compress, encrypt and sign, in the order the bootloader expects:

```python
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from pygbl import GBL3Compression

private_key = load_pem_private_key(
    pathlib.Path("vendor_sign.key").read_bytes(), password=None
)
key = bytes.fromhex("7F8FE53979B31BC556FCB131AFF42414")

sealed = image.compress(GBL3Compression.LZMA).encrypt(key).sign(private_key)
pathlib.Path("signed.gbl").write_bytes(sealed.serialize())
```

The bootloader decompresses into fixed buffers, so `compress` uses the only LZMA
parameters it can accept. Overriding them past what it can allocate raises `ValueError`.

And unwrap it again:

```python
assert sealed.verify_signature(private_key.public_key())
assert sealed.decrypt(key).decompress().serialize() == image.serialize()
```

## Building images from an ELF

Images can be built straight from a linked ELF, without `commander`. Program data comes
from the loadable segments (keyed by *physical* address, since initialized data is
stored in flash but linked at its RAM address) and the application info tag is read
from the SDK's `application_properties_t` struct:

```python
from pygbl import build_application_gbl3, build_bootloader_gbl3

with open("zigbee_ncp.out", "rb") as f:
    image = build_application_gbl3(f, metadata=b'{"fw_type": "zigbee_ncp"}')

with open("bootloader.out", "rb") as f:
    bootloader = build_bootloader_gbl3(f)
```

## GBLv4

Series 3 parts use GBLv4, a different format that nests tags inside a signed manifest
and can bundle several updates in one file.

```python
from pygbl import GBL4Image, GBL4MemorySectionInfo, GBL4UpdateMemorySection

data = pathlib.Path("light-simg301.gbl4").read_bytes()
image, trailing = GBL4Image.deserialize(data)

assert image.serialize() + trailing == data

# `get_tags` searches the whole tree, at any depth
for update in image.get_tags(GBL4UpdateMemorySection):
    print(f"{update.target_address:#010x} {update.plain_image_size} bytes")

for info in image.get_tags(GBL4MemorySectionInfo):
    print(info.compression_scheme, info.encryption_scheme, info.nonce.hex())
```

The manifest binds the rest of the file together, and those bindings can be checked
before flashing anything:

```python
assert image.verify_content_hash()     # the payload after the manifest
assert image.verify_memory_sections()  # each section against the update that names it
assert image.verify_signature(vendor_key)  # the manifest, against a key you trust
```

`verify_signature` needs the key passed in. The image's own `CERTIFICATE` tag is not
used, because anyone who edits the manifest can swap the certificate and re-sign.
Philips Hue images carry no certificate at all.

Sections expand to the image the device flashes:

```python
for section in image.get_tags(GBL4MemorySection):
    print(len(image.plain_image(section)))
```

An encrypted section raises `ValidationError`: the key lives in the device, so the blob
cannot be expanded off-device. `verify_memory_sections` still checks such a section
against the digest the manifest holds for it.

Reading and writing are supported; building a v4 image from an ELF is not, the ELF
helpers are GBLv3 only.

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
