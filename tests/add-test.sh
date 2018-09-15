#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [[ "$#" -eq 0 ]]; then
    ANY=0
    for T in $DIR/orig/*.c; do
        if [[ ! -e $DIR/irix-g/$(basename "$T" .c).s ]]; then
            echo Adding $(basename "$T")
            $DIR/add-test.sh "$T"
            ANY=1
        fi
    done
    if [[ $ANY = 0 ]]; then
        echo "No new tests in tests/orig/."
        echo "Use '$0 tests/orig/<name>.c' to update an existing test." >&2
    else
        echo "Remember to ./run-tests.sh!"
    fi
    exit 0
fi

if [[ "$#" -ne 1 ]] || [[ "$1" != *.c ]]; then
    echo "Usage: $0 [tests/orig/<name>.c]" >&2
    exit 1
fi

INFILE="$1"
BASENAME="$(basename "$INFILE" .c)"

if [ -z "$QEMU_IRIX" ]; then
    echo "env variable QEMU_IRIX should point to the qemu-mips binary" >&2
    exit 2
fi

if [ -z "$IRIX_ROOT" ]; then
    echo "env variable IRIX_ROOT should point to the IRIX compiler directory" >&2
    exit 2
fi

if [ -z "$SM64_TOOLS" ]; then
    echo "env variable SM64_TOOLS should point to a checkout of https://github.com/queueRAM/sm64tools/, with mipsdisasm built" >&2
    exit 2
fi

TEMP="$DIR/../.temp"
declare -A ALL_FLAGS=( ["irix-g"]="-g" ["irix-o2"]="-O2")
WRITTEN=""

for KEY in "${!ALL_FLAGS[@]}"; do
    OUTFILE="$DIR/$KEY/$BASENAME.s"
    SIMPLE_OUTFILE="tests/$KEY/$BASENAME.s"
    FLAGS="${ALL_FLAGS[$KEY]}"
    "$QEMU_IRIX" -silent -L "$IRIX_ROOT" "$IRIX_ROOT/usr/bin/cc" -c -Wab,-r4300_mul -non_shared -G 0 -Xcpluscomm -fullwarn -wlint -woff 819,820,852,821 -signed -DVERSION_JP=1 -mips2 -o $TEMP.o "$INFILE" "$FLAGS"
    mips-linux-gnu-ld -o $TEMP.out $TEMP.o -e test
    LINE=$(mips-linux-gnu-readelf -S $TEMP.out | grep ' \.text')
    ADDR=0x$(echo "$LINE" | tail -c +42 | head -c 8)
    INDEX=0x$(echo "$LINE" | tail -c +51 | head -c 6)
    SIZE=0x$(echo "$LINE" | tail -c +58 | head -c 6)
    ENTRY=$(mips-linux-gnu-readelf -h $TEMP.out | grep Entry | rev | cut -d' ' -f1 | rev)
    "$SM64_TOOLS/mipsdisasm" $TEMP.out $ADDR:$INDEX+$SIZE >"$OUTFILE"
    python3 "$DIR/fixup-lohi.py" "$OUTFILE" "$ENTRY"
    rm $TEMP.o $TEMP.out
    WRITTEN="$WRITTEN $SIMPLE_OUTFILE"
done
echo "Wrote$WRITTEN"
