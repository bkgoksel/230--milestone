#!/bin/bash
set -eu -o pipefail

cl work "$(cat cl/cl_worksheet.txt)"
while true
do
  cl ls
  sleep 5
done
