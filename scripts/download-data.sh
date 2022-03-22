#!/usr/bin/env bash

set -e


if [[ ! -d "data" ]]; then
	mkdir data

	# for the licenses of the downloaded please refer to the README in the downloaded folder
	wget https://www.dropbox.com/sh/t5jw5vbwg8gaoxt/AAA-oQ9FMp2a_Ou7JuhOMiVca?dl=0 -O data.zip
	unzip -d data data.zip -x /

	rm data.zip
fi

