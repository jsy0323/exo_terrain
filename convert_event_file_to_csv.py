#!/usr/bin/env python3
"""Extract the first 2000 Train/mean_reward steps from an event file into CSV."""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterator


def iter_tfrecord_records(path: Path) -> Iterator[bytes]:
    """Yield raw TFRecord payloads from an event file."""
    with path.open("rb") as handle:
        while True:
            header = handle.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError("Unexpected EOF while reading TFRecord length.")

            length = struct.unpack("<Q", header)[0]
            if len(handle.read(4)) != 4:
                raise ValueError("Unexpected EOF while reading TFRecord length CRC.")

            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError("Unexpected EOF while reading TFRecord payload.")

            if len(handle.read(4)) != 4:
                raise ValueError("Unexpected EOF while reading TFRecord data CRC.")

            yield payload


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Unexpected EOF while reading varint.")
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("Varint is too large.")


def iter_fields(buf: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    pos = 0
    while pos < len(buf):
        key, pos = read_varint(buf, pos)
        field_number = key >> 3
        wire_type = key & 0x07

        if wire_type == 0:
            value, pos = read_varint(buf, pos)
            yield field_number, wire_type, value
        elif wire_type == 1:
            end = pos + 8
            if end > len(buf):
                raise ValueError("Unexpected EOF while reading fixed64.")
            yield field_number, wire_type, buf[pos:end]
            pos = end
        elif wire_type == 2:
            length, pos = read_varint(buf, pos)
            end = pos + length
            if end > len(buf):
                raise ValueError("Unexpected EOF while reading length-delimited field.")
            yield field_number, wire_type, buf[pos:end]
            pos = end
        elif wire_type == 5:
            end = pos + 4
            if end > len(buf):
                raise ValueError("Unexpected EOF while reading fixed32.")
            yield field_number, wire_type, buf[pos:end]
            pos = end
        else:
            raise ValueError(f"Unsupported wire type: {wire_type}")


def parse_tensor_scalar(buf: bytes) -> float | None:
    """Best-effort parser for scalar TensorProto values."""
    dtype = None
    tensor_content = None
    float_values: list[float] = []
    double_values: list[float] = []
    int_values: list[int] = []
    int64_values: list[int] = []
    bool_values: list[int] = []
    uint32_values: list[int] = []
    uint64_values: list[int] = []

    for field_number, wire_type, raw_value in iter_fields(buf):
        if field_number == 1 and wire_type == 0:
            dtype = int(raw_value)
        elif field_number == 4 and wire_type == 2:
            tensor_content = bytes(raw_value)
        elif field_number == 5 and wire_type == 2:
            float_values.extend(unpack_packed(raw_value, "<f", 4))
        elif field_number == 6 and wire_type == 2:
            double_values.extend(unpack_packed(raw_value, "<d", 8))
        elif field_number == 7 and wire_type == 2:
            int_values.extend(unpack_packed_varints(raw_value))
        elif field_number == 10 and wire_type == 2:
            int64_values.extend(unpack_packed_varints(raw_value))
        elif field_number == 11 and wire_type == 2:
            bool_values.extend(unpack_packed_varints(raw_value))
        elif field_number == 16 and wire_type == 2:
            uint32_values.extend(unpack_packed_varints(raw_value))
        elif field_number == 17 and wire_type == 2:
            uint64_values.extend(unpack_packed_varints(raw_value))

    if tensor_content and dtype is not None:
        tensor_readers = {
            1: ("<f", 4),
            2: ("<d", 8),
            3: ("<i", 4),
            9: ("<q", 8),
            10: ("<?", 1),
            22: ("<I", 4),
            23: ("<Q", 8),
        }
        reader = tensor_readers.get(dtype)
        if reader and len(tensor_content) >= reader[1]:
            return float(struct.unpack(reader[0], tensor_content[: reader[1]])[0])

    for values in (
        float_values,
        double_values,
        int_values,
        int64_values,
        bool_values,
        uint32_values,
        uint64_values,
    ):
        if values:
            return float(values[0])

    return None


def unpack_packed(buf: bytes, fmt: str, size: int) -> list[float]:
    if len(buf) % size != 0:
        return []
    return [struct.unpack(fmt, buf[i : i + size])[0] for i in range(0, len(buf), size)]


def unpack_packed_varints(buf: bytes) -> list[int]:
    values = []
    pos = 0
    while pos < len(buf):
        value, pos = read_varint(buf, pos)
        values.append(value)
    return values


def parse_summary_value(buf: bytes) -> tuple[str | None, float | None]:
    tag = None
    scalar_value = None

    for field_number, wire_type, raw_value in iter_fields(buf):
        if field_number == 1 and wire_type == 2:
            tag = bytes(raw_value).decode("utf-8", errors="replace")
        elif field_number == 2 and wire_type == 5:
            scalar_value = float(struct.unpack("<f", raw_value)[0])
        elif field_number == 8 and wire_type == 2 and scalar_value is None:
            scalar_value = parse_tensor_scalar(raw_value)

    return tag, scalar_value


def parse_event(buf: bytes) -> tuple[int | None, float | None, list[tuple[str, float]]]:
    step = None
    wall_time = None
    scalars: list[tuple[str, float]] = []

    for field_number, wire_type, raw_value in iter_fields(buf):
        if field_number == 1 and wire_type == 1:
            wall_time = float(struct.unpack("<d", raw_value)[0])
        elif field_number == 2 and wire_type == 0:
            step = int(raw_value)
        elif field_number == 5 and wire_type == 2:
            for nested_field, nested_wire_type, nested_value in iter_fields(raw_value):
                if nested_field != 1 or nested_wire_type != 2:
                    continue
                tag, scalar_value = parse_summary_value(nested_value)
                if tag is not None and scalar_value is not None and math.isfinite(scalar_value):
                    scalars.append((tag, scalar_value))

    return step, wall_time, scalars


TARGET_TAG = "Train/mean_reward"
MAX_STEP = 2000


def export_mean_reward_csv(rows: dict[int, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "mean_reward"])
        writer.writeheader()
        for step in sorted(rows):
            writer.writerow({"step": step, "mean_reward": rows[step]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to an event file such as event.out.0")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path. Defaults to <input>.csv",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_suffix(input_path.suffix + ".csv")

    rows_by_step: dict[int, float] = {}

    for record in iter_tfrecord_records(input_path):
        step, _, scalars = parse_event(record)
        if step is None or not scalars:
            continue
        if step > MAX_STEP:
            continue

        for tag, value in scalars:
            if tag == TARGET_TAG:
                rows_by_step[step] = value
                break

    export_mean_reward_csv(rows_by_step, output_path)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Tag: {TARGET_TAG}")
    print(f"Max step: {MAX_STEP}")
    print(f"Exported rows: {len(rows_by_step)}")
    if rows_by_step:
        steps = sorted(rows_by_step)
        values = [rows_by_step[step] for step in steps]
        print(f"Step range: {steps[0]}..{steps[-1]}")
        print(f"Value range: {min(values):.6g}..{max(values):.6g}")


if __name__ == "__main__":
    main()
