#!/usr/bin/env bash

_script_dir=$(dirname $BASH_SOURCE)
EASYPAPDIR=${EASYPAPDIR:-$(realpath ${_script_dir}/..)}
. ${_script_dir}/easypap-common.bash
unset _script_dir

usage()
{
    echo "Usage: $PROGNAME <kernel name>"
    echo "option can be:" 
    echo "  -h | --help: display help"

    exit $1
}

shall_we_continue()
{
    read -r -p "$1! Are you sure? [y/N] " response
    response=${response,,} # tolower
    if [[ $response =~ ^(y|yes)$ ]] ; then
        return
        #echo "Please do not insist! This is not reasonnable: operation aborted" >&2
        #exit 1
    fi
    echo "Operation aborted" >&2
    exit 1
}

PROGNAME=$0

while [[ $# -ge 1 ]]; do
    case $1 in
        -h|--help)
            usage 0
            ;;
        *)
            break
            ;;
    esac
    shift
done

[[ $# == 1 ]] || usage 1

KERNEL="$1"
CFILE=${EASYPAPDIR}/kernel/c/${KERNEL}.c

if [[ -f $CFILE ]]; then
    shall_we_continue "File $CFILE will be erased"
fi
 
sed -e "s/<template>/$KERNEL/g" < ${EASYPAPDIR}/data/templates/kernel_template.c > $CFILE && echo "Template file $CFILE generated."

exit 0
