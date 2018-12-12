#!/bin/bash
cd "$( dirname "${BASH_SOURCE[0]}" )"

for D in tests/{irix-g,irix-o2}; do
    mkdir -p $D/output/
    for T in $D/*.s; do
        C_FILE=$D/output/$(basename $T .s).c
        FN_NAME=test
        python3 src/main.py --stop-on-error $T $FN_NAME >.stdout 2>.stderr
        if [[ $? != 0 ]]; then
            EXPECTED=$(cat $C_FILE 2>/dev/null)
            echo CRASHED > $C_FILE
            if [[ "$EXPECTED" != "CRASHED" ]]; then
                echo "Unexpected crash for file $T:"
                echo
                cat .stderr
                echo "Quitting."
                rm .stdout .stderr
                exit 1
            fi
        else
            cp .stdout $C_FILE
        fi
    done
done
rm .stderr .stdout
CHANGED="$(git diff --name-only tests/{irix-g,irix-o2}/output/*.c)"
if [[ $CHANGED == '' ]]; then
    echo "No changes."
else
    echo "The following files changed:"
    echo "$CHANGED"
    echo
    echo "Use 'git diff' to inspect the changes."
fi
