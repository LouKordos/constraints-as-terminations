fdfind --type f -H -x sh -c 'printf "\n\n---------------------------File: {}--------------------------\n\n" && cat {}' | curl -F 'file=@-' https://0x0.st
